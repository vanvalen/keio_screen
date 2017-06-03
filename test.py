from keio_names import pos_to_strain, get_keio_names

strain_matrix = get_keio_names()

strain = pos_to_strain(strain_matrix, 9, 'A7')

print strain
