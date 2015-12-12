bpy.context.scene.mcell.cellblender_preferences.decouple_export_run = True
bpy.context.scene.mcell.cellblender_preferences.tab_autocomplete = True

dm_dict = {"data_model_version": "DM_2014_10_24_1638",
           "name": "bla",
           "rxn_name": "bla2",
           "reactants": "mol_1",
           "products": "Mol_2",
           "rxn_type": "reversible",
           "variable_rate_switch": False,
           "variable_rate": "file.txt",
           "variable_rate_valid": True,
           "fwd_rate": 1,
           "bkwd_rate": 2,
           "variable_rate_text": "hello rate" }
mcell.reactions.build_properties_from_data_model(C, dm_dict)
