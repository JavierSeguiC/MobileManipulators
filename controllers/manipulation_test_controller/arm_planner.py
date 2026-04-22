import re
from pathlib import Path
import numpy as np
import datetime
from collections import deque
import math

class Planner:

    def __init__(self,
                planner_limits,
                wbt_paths,
                rrt_q = 0.02,
                rrt_r = 0.01,
                rrt_max_samples = 2000,
                rrt_prc = 0.1,
                headless = True
                ):
        
        self.planner_limits = planner_limits
        self.rrt_q = rrt_q
        self.rrt_r = rrt_r
        self.rrt_max_samples = rrt_max_samples
        self.rrt_prc = rrt_prc
        self.wbt_paths = wbt_paths
        self.headless = headless

        self.VERBOSE = True
        self.FORCE_NO_UNIT_CONVERSION = False
        self.SEARCH_BACK_CHARS = 4000
        self.ALMOST_EQUAL_TOL = 1e-6 

        self.GROUND_NAMES = {'ground', 'floor', 'ground_plane', 'floor_plane', 'groundplane', 'ground-plane'}

        # regext
        self.TRANSLATION_RE = re.compile(r"translation\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")
        self.DEF_RE = re.compile(r'\bDEF\s+([A-Za-z0-9_]+)')
        self.NAME_RE = re.compile(r'name\s+"([^"]+)"')
        
        # Regex for bounding objects (Box, Cylinder, Sphere) in both boundingObject and geometry contexts
        self.BOX_RE = re.compile(r"(?:boundingObject|geometry)\s+Box\s*\{\s*size\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*\}", re.DOTALL | re.IGNORECASE)
        self.CYL_RE = re.compile(r"(?:boundingObject|geometry)\s+Cylinder\s*\{\s*radius\s+([-\d.eE]+)\s+height\s+([-\d.eE]+)\s*\}", re.DOTALL | re.IGNORECASE)
        self.SPH_RE = re.compile(r"(?:boundingObject|geometry)\s+Sphere\s*\{\s*radius\s+([-\d.eE]+)\s*\}", re.DOTALL | re.IGNORECASE)
        
        self.IFS_COORD_RE = re.compile(r"IndexedFaceSet\s*\{[\s\S]*?coord\s+Coordinate\s*\{\s*point\s*\[\s*([^\]]+)\]", re.IGNORECASE)

        self.PARENT_BLOCK_RE = re.compile(r"\b(Solid|Transform|Group|Proto)\b", re.IGNORECASE)
        
        self.ROTATION_RE = re.compile(r"rotation\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")
        self.SCALE_RE = re.compile(r"scale\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")

        # Find webots world file
        wbt_path = self.find_wbt_default()
        if wbt_path is None:
            raise FileNotFoundError(
                f"The file .wbt was not found.\n"
                f"Paths searched: {self.wbt_paths}"
            )

        search_bounds = ((self.planner_limits[0,0], self.planner_limits[0,1]),
                            (self.planner_limits[1,0], self.planner_limits[1,1]),
                            (self.planner_limits[2,0], self.planner_limits[2,1]))

        self.obstacles, self.boxes_info = self.build_bboxes_from_wbt(wbt_path, search_bounds=search_bounds)

        self.debug_print("[main] Obstacles array (for RRT):")
        self.debug_print(self.obstacles)
        self.debug_print("[main] parsed objects detail:")
        for b in self.boxes_info:
            self.debug_print(" -", b.get("name"), b.get("type"), "enclosing:", b.get("enclosing_solid_name"),
                        "size:", b.get("size_scaled"), "translation_abs_used:", b.get("translation_abs_used"),
                        "translation_parts:", b.get("translation_parts"), "bbox:", b.get("bbox"))

        if self.obstacles.size == 0:
            self.debug_print("[main] No valid obstacles found; aborting planner.")
            
    def axis_angle_to_matrix(self, ax, ay, az, angle):
        
        # Normalize
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm < 1e-6:
            return np.eye(3)
        
        ax, ay, az = ax/norm, ay/norm, az/norm
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c
        
        # Rodrigues' formula converted to matrix
        R = np.array([
            [t*ax*ax + c,    t*ax*ay - s*az, t*ax*az + s*ay],
            [t*ax*ay + s*az, t*ay*ay + c,    t*ay*az - s*ax],
            [t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c]
        ])
        return R

    def get_objects_positions(self, object_names):
        """
        Receives a list of object names (strings) and returns a list
        of dictionaries with the name and absolute position [x, y, z].
        Searches both in the geometry name and the containing Solid name.
        """
        results = []
        
        # For each object
        for target_name in object_names:
            found = False
            
            # For each box info parsed from the .wbt, check if the target name matches either the geometry name or the enclosing solid name
            for box in self.boxes_info:
                # Extraemos los dos posibles nombres
                geo_name = box.get("name")
                solid_name = box.get("enclosing_solid_name")
                
                if target_name == geo_name or target_name == solid_name:
                    
                    # Obtain position
                    pos = box.get("translation_abs_used")
                    
                    results.append({
                        "name": target_name,
                        "position": list(pos) if pos is not None else [0.0, 0.0, 0.0]
                    })
                    found = True
                    break
            
            if not found:
                if self.VERBOSE:
                    print(f"[get_objects_positions] Warning: '{target_name}' not found in the .wbt")
                results.append({
                    "name": target_name,
                    "position": None
                })
                
        return results
    
    def now_ts(self):
        return datetime.datetime.now().isoformat(timespec="seconds")


    def debug_print(self, *args, **kwargs):
        if self.VERBOSE:
            print(f"[{self.now_ts()}]", *args, **kwargs)


    def almost_equal_vec(self, a, b, tol=None):
        """True if a and b are 3-element vectors and are within distance <= tol (L_inf)."""

        if tol is None:
            tol = self.ALMOST_EQUAL_TOL

        if a is None or b is None:
            return False
        return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(3))


    def get_solid_blocks(self, text):
        solid_token_re = re.compile(r'\bSolid\b\s*\{', re.IGNORECASE)
        starts = [m.start() for m in solid_token_re.finditer(text)]

        spans = []
        n = len(text)
        for s in starts:
            brace_pos = text.find('{', s)
            if brace_pos == -1: continue
            depth = 0
            i = brace_pos
            while i < n:
                c = text[i]
                if c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        spans.append((s, end_pos))
                        break
                i += 1

        blocks = []
        for idx, (st, ed) in enumerate(spans):
            block_text = text[st:ed]
            name_m = self.NAME_RE.search(block_text)
            name = name_m.group(1) if name_m else None
            
            tr_m = self.TRANSLATION_RE.search(block_text)
            translation = tuple(map(float, tr_m.groups())) if tr_m else (0.0, 0.0, 0.0)
            
            rot_m = self.ROTATION_RE.search(block_text)
            if rot_m:
                rx, ry, rz, ra = map(float, rot_m.groups())
                rotation_matrix = self.axis_angle_to_matrix(rx, ry, rz, ra)
            else:
                rotation_matrix = np.eye(3)
            
            sc_m = self.SCALE_RE.search(block_text)
            scale_vec = tuple(map(float, sc_m.groups())) if sc_m else (1.0, 1.0, 1.0)

            blocks.append({
                "idx": idx,
                "start": st,
                "end": ed,
                "name": name,
                "translation": translation,    
                "rotation": rotation_matrix,    
                "scale": scale_vec,
                "children": [],
                "parent": None,
                "translation_abs": None,
                "rotation_abs": None 
            })

        # Assign inmediate parent-child relationships based on block nesting
        for i, b in enumerate(blocks):
            st, ed = b["start"], b["end"]
            parent_idx = None
            parent_span_size = None
            for j, cand in enumerate(blocks):
                if j == i: continue
                if cand["start"] < st and cand["end"] >= ed:
                    size = cand["end"] - cand["start"]
                    if parent_idx is None or size < parent_span_size:
                        parent_idx = j
                        parent_span_size = size
            b["parent"] = parent_idx
            if parent_idx is not None:
                blocks[parent_idx]["children"].append(i)

        return blocks


    def remove_comments(self, text: str) -> str:
        text = re.sub(r"//.*", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)
        return text


    def find_bounding_objects(self, text: str):

        found = []

        # 1) Boxes (boundingObject o geometry)
        for m in self.BOX_RE.finditer(text):
            sx, sy, sz = [float(x) for x in m.groups()]
            found.append({"type": "Box", "size": [sx, sy, sz], "pos": m.start(), "snippet": text[max(0,m.start()-300):m.end()+300]})

        # 2) Cylinders
        for m in self.CYL_RE.finditer(text):
            radius, height = [float(x) for x in m.groups()]
            found.append({"type": "Cylinder", "size": [radius, height], "pos": m.start(), "snippet": text[max(0,m.start()-300):m.end()+300]})

        # 3) Spheres
        for m in self.SPH_RE.finditer(text):
            radius = float(m.group(1))
            found.append({"type": "Sphere", "size": [radius], "pos": m.start(), "snippet": text[max(0,m.start()-300):m.end()+300]})

        # 4) IndexedFaceSet -> extract points and convert to AABB (Box)
        for m in self.IFS_COORD_RE.finditer(text):
            pts_block = m.group(1)

            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", pts_block)
            if not nums:
                continue
            vals = [float(x) for x in nums]
            
            if len(vals) < 3:
                continue
            ntrip = (len(vals) // 3)
            coords = np.array(vals[:ntrip*3]).reshape((-1,3))
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            center = (mins + maxs) / 2.0
            size = (maxs - mins).tolist()

            eps = 1e-6
            size = [s if s > eps else eps for s in size]
            found.append({"type": "Box", "size": size, "pos": m.start(), "snippet": text[max(0,m.start()-300):m.end()+300], "center_from_coords": tuple(center.tolist())})

        found.sort(key=lambda x: x['pos'])
        self.debug_print(f"Found {len(found)} bounding/geometry (including IFS).")
        return found


    def find_nearest_translation_before(self, text: str, pos: int, window=None):
        
        if window is None:
            window = self.SEARCH_BACK_CHARS
        
        start = max(0, pos - window)
        snippet = text[start:pos]
        matches = list(self.TRANSLATION_RE.finditer(snippet))
        if not matches:
            return None
        m = matches[-1]
        tx, ty, tz = [float(x) for x in m.groups()]
        abspos = start + m.start()
        return (tx, ty, tz, abspos)


    def find_parent_block_pos_before(self, text: str, pos: int, window=None):
        
        if window is None:
            window = self.SEARCH_BACK_CHARS
        
        start = max(0, pos - window)
        snippet = text[start:pos]
        matches = list(self.PARENT_BLOCK_RE.finditer(snippet))
        if not matches:
            return None
        m = matches[-1]
        return start + m.start()


    def find_translation_after_pos(self, text: str, start_pos: int, stop_pos: int):
        
        if start_pos is None:
            return None
        
        if start_pos < 0:
            start_pos = 0
        
        snippet = text[start_pos:stop_pos]
        m = self.TRANSLATION_RE.search(snippet)
        
        if not m:
            return None
        
        tx, ty, tz = [float(x) for x in m.groups()]
        abspos = start_pos + m.start()
        return (tx, ty, tz, abspos)

    def find_name_near(self, text: str, pos: int, parent_token_pos: int = None, window_pre=800, window_post=800):
        total_len = len(text)
        start = max(0, pos - window_pre)
        end = min(total_len, pos + window_post)
        snippet = text[start:end]
        m = self.NAME_RE.search(snippet)
        if m:
            return m.group(1)
        m2 = self.DEF_RE.search(snippet)
        if m2:
            return m2.group(1)
        ext_end = min(total_len, pos + 2000)
        if ext_end > end:
            snippet2 = text[end:ext_end]
            m = self.NAME_RE.search(snippet2)
            if m:
                return m.group(1)
            m2 = self.DEF_RE.search(snippet2)
            if m2:
                return m2.group(1)
        if parent_token_pos is not None and 0 <= parent_token_pos < pos:
            snippet_parent_block = text[parent_token_pos:pos]
            m = self.NAME_RE.search(snippet_parent_block)
            if m:
                return m.group(1)
            m2 = self.DEF_RE.search(snippet_parent_block)
            if m2:
                return m2.group(1)
        deep_start = max(0, pos - 3000)
        if deep_start < start:
            deep_snip = text[deep_start:start]
            m2 = self.DEF_RE.search(deep_snip)
            if m2:
                return m2.group(1)
        return None

    def detect_scale_from_records(self, records):
        
        if self.FORCE_NO_UNIT_CONVERSION:
            self.debug_print("FORCE_NO_UNIT_CONVERSION=True -> scale=1.0")
            return 1.0
        
        vals = []
        
        for r in records:
            tr = r.get('translation_raw_parent') or (None,None,None)
            if tr[0] is not None:
                vals.extend([abs(x) for x in tr])
            tc = r.get('translation_raw_child') or (None,None,None)
            if tc[0] is not None:
                vals.extend([abs(x) for x in tc])
            s = r.get('size_raw') or []
            vals.extend([abs(x) for x in s])
        if not vals:
            return 1.0
        maxv = max(vals)
        
        if maxv > 1000.0:
            self.debug_print(f"Heuristics: max value {maxv} > 1000 -> applying scale=0.01 (cm->m)")
            return 0.01
        self.debug_print(f"Heuristics: max value {maxv} -> scale = 1.0")
        return 1.0

    def clamp_warn_bbox(self, bbox, bounds):
        (xmin_b, xmax_b), (ymin_b, ymax_b), (zmin_b, zmax_b) = bounds
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        outside = False
        if xmin < xmin_b or xmax > xmax_b or ymin < ymin_b or ymax > ymax_b or zmin < zmin_b or zmax > zmax_b:
            outside = True
        if outside:
            self.debug_print(f"[clamp_warn_bbox] WARNING: bbox {bbox} out of bounds {bounds} — keeping bbox as is.")
        return bbox

    # ---------------- compute absolute translations for solids ----------------
    def compute_absolute_transforms(self, solids, scale=1.0):
        
        roots = [s['idx'] for s in solids if s['parent'] is None]
        

        for r in roots:

            t_loc = np.array(solids[r]['translation']) * scale
            r_loc = solids[r]['rotation']
            
            solids[r]['translation_abs'] = t_loc
            solids[r]['rotation_abs'] = r_loc

        q = deque(roots)
        
        while q:
            cur_idx = q.popleft()
            cur_block = solids[cur_idx]
            
            parent_t_abs = cur_block['translation_abs']
            parent_r_abs = cur_block['rotation_abs']
            
            for cidx in cur_block['children']:
                child_block = solids[cidx]
                

                t_child_local = np.array(child_block['translation']) * scale
                r_child_local = child_block['rotation']
                
                child_r_abs = parent_r_abs @ r_child_local
                t_child_rotated = parent_r_abs @ t_child_local
                child_t_abs = parent_t_abs + t_child_rotated
                
                # Guardamos resultados
                solids[cidx]['translation_abs'] = child_t_abs
                solids[cidx]['rotation_abs'] = child_r_abs
                
                q.append(cidx)

        for s in solids:
            if s['translation_abs'] is None:
                s['translation_abs'] = np.array(s['translation']) * scale
            if s['rotation_abs'] is None:
                s['rotation_abs'] = np.eye(3)


    def build_bboxes_from_wbt(self, path_wbt: Path, search_bounds = ((-1,1),(-1,1),(0,1.5))):
        if not path_wbt.exists():
            raise FileNotFoundError(f"{path_wbt} no existe")
        text = path_wbt.read_text(encoding="utf-8")
        text = self.remove_comments(text)

        solid_blocks = self.get_solid_blocks(text)
        b_objs = self.find_bounding_objects(text)
        
        self.debug_print(f"Detectados {len(solid_blocks)} bloques Solid (idx, name, translation):",
                [(b["idx"], b["name"], b["translation"]) for b in solid_blocks])
        
        records = []
        for bo in b_objs:

            pos = bo['pos']
            parent_token_pos = self.find_parent_block_pos_before(text, pos, window=self.SEARCH_BACK_CHARS)
            name = self.find_name_near(text, pos, parent_token_pos=parent_token_pos)
            child_tr = self.find_nearest_translation_before(text, pos, window=800)
            parent_tr = None
            if parent_token_pos is not None:
                parent_tr = self.find_translation_after_pos(text, parent_token_pos, pos)
            records.append({
                "name": name,
                "type": bo['type'],
                "size_raw": bo['size'],
                "pos_hint_center": bo.get('center_from_coords'),
                "translation_raw_child": child_tr[:3] if child_tr else None,
                "translation_child_pos": child_tr[3] if child_tr else None,
                "translation_raw_parent": parent_tr[:3] if parent_tr else None,
                "translation_parent_pos": parent_tr[3] if parent_tr else None,
                "parent_token_pos": parent_token_pos,
                "bounding_pos": pos,
                "snippet": bo['snippet']
            })


        scale = self.detect_scale_from_records(records)
        
        self.compute_absolute_transforms(solid_blocks, scale=scale)


        boxes = []
    
        def find_enclosing_solid_idx(pos):
            candidates = []
            for b in solid_blocks:
                if b["start"] <= pos < b["end"]:
                    size = b["end"] - b["start"]
                    candidates.append((size, b["idx"]))
            if not candidates:
                return None
            candidates.sort()
            return candidates[0][1]

        for r in records:
            name = r.get("name")
            typ = r.get("type")
            size_raw = r.get("size_raw")
            pos_hint_center = r.get("pos_hint_center")  # center if created from IFS
            t_child = r.get("translation_raw_child")
            t_parent = r.get("translation_raw_parent")
            parent_token_pos = r.get("parent_token_pos")
            bounding_pos = r.get("bounding_pos")

            t_child_s = None if t_child is None else [float(x)*scale for x in t_child]
            t_parent_s = None if t_parent is None else [float(x)*scale for x in t_parent]
            size = None if size_raw is None else [float(x)*scale for x in size_raw]

            # Ignore floor
            if name is not None and name.strip().lower() in self.GROUND_NAMES:
                self.debug_print(f"Ignorando objeto '{name}' (suelo).")
                continue

            enclosing_idx = find_enclosing_solid_idx(bounding_pos)
            enclosing_block = None if enclosing_idx is None else solid_blocks[enclosing_idx]
            

            if enclosing_block:
                parent_R = enclosing_block.get('rotation_abs', np.eye(3))
                parent_T = np.array(enclosing_block.get('translation_abs', [0,0,0]))
            else:
                parent_R = np.eye(3)
                parent_T = np.array([0., 0., 0.])

            if enclosing_block:
                self.debug_print(f"Bounding pos {bounding_pos} inside Solid '{enclosing_block['name']}'")

            parts_used = []
            final_pos = np.array([0., 0., 0.])
            

            vec_to_np = lambda v: np.array(v) if v is not None else np.zeros(3)

            if enclosing_block is not None and t_child_s is not None:

                enclosing_abs_tuple = tuple(parent_T)
                if self.almost_equal_vec(t_child_s, enclosing_abs_tuple):

                    final_pos = vec_to_np(t_child_s)
                    parts_used.append(('child_local (equal_to_enclosing_abs, no_sum)', tuple(t_child_s)))
                else:
                    offset_rotated = parent_R @ vec_to_np(t_child_s)
                    final_pos = parent_T + offset_rotated
                    
                    parts_used.append(('enclosing_abs', tuple(parent_T)))
                    parts_used.append(('child_local_rotated', tuple(offset_rotated)))


            elif enclosing_block is not None and pos_hint_center is not None:
                if self.almost_equal_vec(pos_hint_center, (0.0,0.0,0.0)):
                    final_pos = parent_T
                    parts_used.append(('enclosing_abs', tuple(parent_T)))
                else:

                    center_local = vec_to_np(pos_hint_center) * scale
                    offset_rotated = parent_R @ center_local
                    final_pos = parent_T + offset_rotated
                    
                    parts_used.append(('enclosing_abs', tuple(parent_T)))
                    parts_used.append(('ifs_center_rotated', tuple(offset_rotated)))

            # Caso 3: Solo padre
            elif enclosing_block is not None:
                final_pos = parent_T
                parts_used.append(('enclosing_abs', tuple(parent_T)))

            # Caso 4: Sin padre, usando child local (raro, sería absoluto)
            elif t_child_s is not None:
                final_pos = vec_to_np(t_child_s)
                parts_used.append(('child_local', tuple(t_child_s)))

            # Caso 5: Usando parent raw (fallback)
            elif t_parent_s is not None:
                final_pos = vec_to_np(t_parent_s)
                parts_used.append(('parent_raw', tuple(t_parent_s)))

            else:
                self.debug_print(f"Ignoro objeto '{name}' (no translations detectadas).")
                continue

            # Extraer coordenadas finales para uso posterior
            tx, ty, tz = final_pos

            # --- CÁLCULO DE BOUNDING BOX ROTADA (AABB) ---
            
            # size: default si viene de IFS
            if size is None and pos_hint_center is not None:
                size = [0.01, 0.01, 0.01]

            if size is None:
                self.debug_print(f"Objeto '{name}' tipo {typ} sin size -> omitido")
                continue

            parent_scale = np.array(enclosing_block.get('scale', (1.0, 1.0, 1.0))) if enclosing_block else np.array([1.0, 1.0, 1.0])

            # 1. Definir las dimensiones medias (half-extents) APLICANDO ESCALA
            if typ == 'Box':
                # Multiplicamos elemento a elemento: size * scale
                real_size = np.array(size) * parent_scale
                half_extents = real_size / 2.0
                
            elif typ == 'Cylinder':
                # El cilindro escala radio (x, y) y altura (z)
                # Ojo: scale[0] afecta al radio, scale[2] a la altura
                r_scaled = size[0] * parent_scale[0] 
                h_scaled = size[1] * parent_scale[2]
                half_extents = np.array([r_scaled, r_scaled, h_scaled/2.0])
                
            elif typ == 'Sphere':
                # Esfera escala uniforme (usamos X como referencia)
                r_scaled = size[0] * parent_scale[0]
                half_extents = np.array([r_scaled, r_scaled, r_scaled])
            else:
                half_extents = np.array([0.05, 0.05, 0.05])

            # 2. Generar las 8 esquinas de la caja centradas en (0,0,0) local
            sx, sy, sz = half_extents
            corners = np.array([
                [ sx,  sy,  sz], [ sx,  sy, -sz], [ sx, -sy,  sz], [ sx, -sy, -sz],
                [-sx,  sy,  sz], [-sx,  sy, -sz], [-sx, -sy,  sz], [-sx, -sy, -sz]
            ])

            # 3. Rotar las esquinas usando la matriz del padre
            # (Asumimos que el BoundingObject está alineado con el Solid padre)
            corners_rotated = (parent_R @ corners.T).T 

            # 4. Encontrar min y max de las esquinas rotadas
            min_corner = np.min(corners_rotated, axis=0)
            max_corner = np.max(corners_rotated, axis=0)

            # 5. Sumar la posición central final calculada antes
            final_min = final_pos + min_corner
            final_max = final_pos + max_corner

            bbox = (final_min[0], final_min[1], final_min[2], 
                    final_max[0], final_max[1], final_max[2])
            
            bbox_warned = self.clamp_warn_bbox(bbox, search_bounds)

            boxes.append({
                "name": name,
                "type": typ,
                "size_scaled": size,
                "translation_parts": parts_used,
                # IMPORTANTE: Guardamos la posición calculada (vector numpy convertido a tupla)
                "translation_abs_used": tuple(final_pos),
                "enclosing_solid_idx": enclosing_idx,
                "enclosing_solid_name": enclosing_block['name'] if enclosing_block else None,
                "bbox": bbox_warned,
                "snippet": r['snippet'][:200]
            })
            self.debug_print(f" -> final Rotated BBox for '{name}': {bbox_warned} (Center: {final_pos})")

        if not boxes:
            self.debug_print("No se construyeron boxes desde el .wbt.")
            return np.zeros((0,6)), []
        arr = np.array([b['bbox'] for b in boxes], dtype=float)
        return arr, boxes

    # ---------------- RRT integration (sin cambios) ----------------
    def run_rrt_with_obstacles(self,
                            x_init,
                            x_goal,
                            planner_limits = None,
                            q = None, r = None, max_samples = None, prc = None,headless=True):
        
        if planner_limits is None:
            planner_limits = self.planner_limits
        
        if q is None:
            q = self.rrt_q
        
        if r is None:
            r = self.rrt_r
        
        if max_samples is None:
            max_samples = self.rrt_max_samples
        
        if prc is None:
            prc = self.rrt_prc

        if headless is None:
            headless = self.headless
        
        self.debug_print("[run_rrt] intentando importar rrt_algorithms...")
        try:
            from rrt_algorithms.rrt.rrt import RRT
            from rrt_algorithms.search_space.search_space import SearchSpace
            from rrt_algorithms.utilities.plotting import Plot
        except Exception as e:
            self.debug_print("[run_rrt] ERROR importando rrt_algorithms:", e)
            return None

        self.debug_print("[run_rrt] creando SearchSpace con dimensions:", planner_limits.tolist())
        X = SearchSpace(planner_limits, self.obstacles)
        self.debug_print("[run_rrt] SearchSpace creado.")
        self.debug_print("[run_rrt] inicializando RRT (q=%s r=%s max=%s prc=%s)" % (q, r, max_samples, prc))
        rrt = RRT(X, q, x_init, x_goal, max_samples, r, prc)

        self.debug_print("[run_rrt] ejecutando rrt_search() ...")
        try:
            path = rrt.rrt_search()
            self.debug_print("[run_rrt] rrt_search completado.")
        except Exception as e:
            self.debug_print("[run_rrt] ERROR en rrt_search():", e)
            return None

        self.debug_print("[run_rrt] Path encontrado:", path)
        if headless==False:
            try:
                plot = Plot("rrt_3d")
                plot.plot_tree(X, rrt.trees)
                if path is not None:
                    plot.plot_path(X, path)
                if self.obstacles.size != 0:
                    plot.plot_obstacles(X, self.obstacles)
                plot.plot_start(X, x_init)
                plot.plot_goal(X, x_goal)
                plot.draw(auto_open=True)
                self.debug_print("[run_rrt] plotting done")
            except Exception as e:
                self.debug_print("[run_rrt] plotting falló (headless?):", e)

        return path


    def find_wbt_default(self):
        
        self.debug_print("[main] inicio: parsing .wbt -> generar Obstacles -> ejecutar RRT")
        
        for p in self.wbt_paths:
            if p.exists():
                self.debug_print("[main] usando .wbt:", p)
                return p
        p = Path.cwd().resolve()
        for _ in range(6):
            for f in p.glob("*.wbt"):
                self.debug_print("[main] usando .wbt:", f)
                return f
            p = p.parent


        self.debug_print("[main] No encontré ManipulationStage.wbt. Pon la ruta en DEFAULT_WBT_PATHS o en cwd.")

        return None

