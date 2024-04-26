import bpy

def print_indent(depth):
    print('    ' * depth, end="")

def parse_material(material):
    
    print('material name:', material.name)

#    print(material.node_tree.__class__)
#    print(material.node_tree.type)

    nodes = material.node_tree.nodes

    index = 0;
    for (i, node) in enumerate(nodes):
        if node.type == "OUTPUT_MATERIAL":
            index = i
            break

    material_output = nodes[index]
    print(material_output.type, material_output.__class__)
    print('target:', material_output.target)

    print('inputs:')

    for input in material_output.inputs:
        if len(input.links) > 0:
            assert(len(input.links) == 1)
            parse_node(input.links[0].from_node, depth=1)

def parse_group(group, depth):
    
#    print_indent(depth)
#    print('group name:', group.name)

    nodes = group.node_tree.nodes

    index = 0;
    for (i, node) in enumerate(nodes):
        if node.type == "GROUP_OUTPUT":
            index = i
            break

    group_output = nodes[index]

    print_indent(depth)
    print(group_output.type, group_output.__class__)
    print_indent(depth)
    print('inputs:')

    for input in group_output.inputs:
        if len(input.links) > 0:
            assert(len(input.links) == 1)
            parse_node(input.links[0].from_node, depth=depth + 1)


def parse_node(node, depth):
    print_indent(depth)
    print(node.type, node.__class__)
    
    def scan_input(node, depth):
        if len(node.inputs) > 0:
            print_indent(depth)
            print('inputs:')

        for input in node.inputs:
            print_indent(depth + 1)
            print(input.name, end="")
            if len(input.links) > 0:
                assert(len(input.links) == 1)
                print(":")
                parse_node(input.links[0].from_node, depth=depth + 2)
            else:
                print(f": [{input.type}] ", end="")
                match input.type:
                    case "VALUE":
                        print(input.default_value)
                    case "RGBA":
                        color = input.default_value
                        print(color[0], color[1], color[2], color[3])
                    case "VECTOR":
                        vector = input.default_value
                        print(vector[0], vector[1], vector[2])
                    case _:
                        print("INPUT TYPE", input.type, "NOT IMPLEMENTED YET",  input.default_value)
    
    match node.type:
        case "ATTRIBUTE":
            print_indent(depth)
            print("attribute type:", node.attribute_type)
            print_indent(depth)
            print("attribute name:", node.attribute_name)
            scan_input(node, depth)
        
        case "MIX_SHADER":
            scan_input(node, depth)

        case "BSDF_DIFFUSE":
            scan_input(node, depth)
        case "BSDF_GLOSSY":
            print_indent(depth)
            print('distribution:', node.distribution)
            scan_input(node, depth)
        
        case "MIX":
            print_indent(depth)
            print("data type:", node.data_type)
            print_indent(depth)
            print("blend type:", node.blend_type)
            print_indent(depth)
            print("clamp result:", node.clamp_result)
            print_indent(depth)
            print("clamp factor", node.clamp_factor)
            scan_input(node, depth)
        
        case "TEX_IMAGE":
            print_indent(depth)
            print("interpolation:", node.interpolation)
            print_indent(depth)
            print("projection:", node.projection)
            print_indent(depth)
            print("extension:", node.extension)

#            print("color mapping:", node.color_mapping)
            print_indent(depth)
            print("image:", node.image.filepath)
#            print(node.image_user)
#            print(node.texture_mapping)

            scan_input(node, depth)

        case "DISPLACEMENT":
            scan_input(node, depth)
        case "INVERT":
            scan_input(node, depth)

        case "MATH":
            print_indent(depth)
            print("operation:", node.operation)
            print_indent(depth)
            print("clamp:", node.use_clamp)
            scan_input(node, depth)

        case "GROUP":
            parse_group(node, depth)

        case _:
            print_indent(depth)
            print("NODE TYPE", node.type, "NOT IMPLEMENTED YET")
            

obj = bpy.data.objects['Brick_flat_02*08.002']

material_slots = obj.material_slots

for material_slot in material_slots:
    material = material_slot.material
    parse_material(material)
