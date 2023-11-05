import zarr
from zarr.conv
import h5py

# TODO: refactor var name latent_code -> data
def save_data_to_hdf(latent_code:torch.Tensor, metadata:dict, method:str, root_dir: str):
    case_id = metadata['case_id']
    stain_id = metadata['stain_id']
    location = metadata['location']

    for recon, latent, loc, stain, case in zip(latent_code, location, stain_id, case_id):
        location = location.replace(",", "-").strip('()')
        filename = f"{root_dir}/{case_id}" + (f"_{stain_id}_" if method == "case" else "") + "_latents.hdf5"

        with h5py.File(filename, "a") as file:
            file.attrs.update(metadata)
            
                match method:
                    case "case":
                        #  one nesting level
                        prefix = f"{case_id}_{stain_id}"
                        filename = prefix + ".zarr"

                            if location not in file:
                                file.create_group(location)
                            file[location].create_dataset(location, data=latent_code)

                    case "stain":
                        # two nesting levels
                        # case
                        #     |__ stain
                        #              |__ location
                        prefix = f"{case_id}"
                        filename = prefix + ".zarr"
                            
                            if stain not in file:
                                file.create_group(stain)
                            elif location not in file[stain]:
                                file[stain].create_group(location)
                            file[stain][location].create_dataset(location, data=latent_code)

def save_to_zarr(reconstructed_img, latent_code, metadata, method, root_dir):

    case_id = metadata['case_id']
    stain_id = metadata['stain_id']
    location = metadata['location']

    match method:
    case "case":
        prefix = f"{case_id}_{stain_id}"
        filename = prefix + ".zarr"
            for recon, latent, loc, stain, case in zip(reconstructed_imd, latent_code, location, stain_id, case_id):
                filename = f"{case}.zarr"
                z = zarr.open(filename, mode='a')
                z.attrs.update(metadata) # FIXME: does this work?
                # this should be equivalent to open(), but automatically create subgroups as well
                z.open_group(f'{case}/{stain}/{loc}', mode='a')
                # z, is_new = get_or_create_zarr(filename)
                # this should create or open
                g = open_group(f'{case}/{stain}/{loc}', mode="a")
                # each location should be unique so alway creating a new arr should be fine

                g.create_dataset('reconstructed', data=np.random((3,4,5)), chunks=False)
                g.create_dataset('latent_code', data=np.random((1,4,5)), chunks=False)

                # if is_new:
                #     g_s = z.create_group(stain)
                #     g_l = g_s.create_group(loc)
                #     g_l.create_dataset('reconstructed', data=sample_data['reconstructed'], chunks=(1,) + sample_data['reconstructed'].shape[1:])
                #     g_l.create_dataset('latent_code', data=sample_data['latent_code'], chunks=(1,) + sample_data['latent_code'].shape[1:])
                # else:
                #     g = z[stain][loc]
                #     g['reconstructed'].append(sample_data['reconstructed'])
                #     g['latent_code'].append(sample_data['latent_code'])
    
    case "stain":
        prefix = f"{case_id}_{stain_id}"
        filename = prefix + ".zarr"
            for recon, latent, loc, stain, case in zip(reconstructed_imd, latent_code, location, stain_id, case_id):
                filename = f"{case_id}.zarr"
                z, is_new = get_or_create_zarr(filename)
                # this should create or open
                g = open_group(f'{case}/{stain}/{loc}', mode="a")
                # each location should be unique so alway creating a new arr should be fine
                g.create_dataset('reconstructed', data=sample_data['reconstructed'], )
                g.create_dataset('latent_code', data=sample_data['latent_code'], )


        prefix = f"{case_id}"
        filename = prefix + ".zarr"
        if os.path.isfile(filename):
            z = zarr.open(filename, mode='r+')
            for loc, sample_data in data.items():
                g = z[loc]
                g['reconstructed'].append(sample_data['reconstructed'])
                g['latent_code'].append(sample_data['latent_code'])
        
        # else:
        #     z = zarr.group(store=zarr.DirectoryStore(filename))
        #     z.attrs.update(metadata)

        #     for loc, sample_data in data.items():
        #         g = z.create_group(loc)
        #         g.create_dataset('reconstructed', data=sample_data['reconstructed'], chunks=(1,) + sample_data['reconstructed'].shape[1:])
        #         g.create_dataset('latent_code', data=sample_data['latent_code'], chunks=(1,) + sample_data['latent_code'].shape[1:])
        # ##########################
        def get_or_create_zarr(filename)
            if os.path.isfile(filename):
                z = zarr.open(filename, mode='r+')
                return z, False
            else:
                z = zarr.group(store=zarr.DirectoryStore(filename))
                z.attrs.update(metadata)
                return z, True
