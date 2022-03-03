import sys
import bpy 

for i in bpy.data.objects:
    bpy.data.objects.remove(i, do_unlink=True)

# gar_loc = "./pred_gar_0000.obj"
gar_loc = sys.argv[1]
# tex_loc = "./pred_gar_0000.jpg"
tex_loc = sys.argv[2]

bpy.ops.import_scene.obj(filepath=gar_loc)

mesh = bpy.context.selected_objects[0]  #### Imported objected gets id 0

bpy.context.view_layer.objects.active = mesh
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode='OBJECT')
#--------------------

mat = bpy.data.materials.new(name='mat')
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
texImage.image = bpy.data.images.load(tex_loc)
mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

# Assign it to object
if mesh.data.materials:
    mesh.data.materials[0] = mat
else:
    mesh.data.materials.append(mat)


#--------------------
# bpy.ops.export_scene.obj(filepath="", check_existing=True, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_selection=False, use_animation=False, use_mesh_modifiers=True, use_edges=True, use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True, use_uvs=True, use_materials=True, use_triangles=False, use_nurbs=False, use_vertex_groups=False, use_blen_objects=True, group_by_object=False, group_by_material=False, keep_vertex_order=False, global_scale=1, path_mode='AUTO')
bpy.ops.export_scene.obj(filepath=gar_loc)
