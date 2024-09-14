import shutil
import os
import argparse
import subprocess
# import debugpy

# debugpy.listen(5679)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()


def copy_directory(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    current_directory = os.getcwd()
    source_dir = os.path.abspath(os.path.join(current_directory, source_dir))
    destination_dir = os.path.abspath(os.path.join(current_directory, destination_dir))
    print(source_dir)
    print(destination_dir)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)


    #if not os.path.exists(destination_dir):
    #    os.makedirs(destination_dir)

    # Copy the contents of the source directory to the destination directory
    try:
        shutil.copy2(source_dir, destination_dir)
        print("Directory copied successfully!")
    except Exception as e:
        print("Error:", e)

def list_of_floats(arg):
    return list(map(str, arg.split(',')))

if __name__ == "__main__":
    # Specify the paths of the source and destination directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--kld_weight", type=str, default="")
    parser.add_argument("--latent_size", type=str, default="")
    parser.add_argument("--scheduled_sampling_length", type=str, default="1")

    args = parser.parse_args()
    #source_directory = "./train_cvae_light_ar_cond_dof2.py"
    #source_directory = "./train_cvae_light_ar_cond_part_wise_dof.py"
    #source_directory = "./train_cvae_light_ar_cond_dof_lowerOnly.py"
    #source_directory = "./train_cvae_light_ar_cond_dof_upperOnly.py"
    #source_directory = "./train_cvae_light_ar_cond_dof_rootOnly.py"
    #source_directory = "./train_wvae_light_ar_cond_dof_rootOnly.py"
    #source_directory = "./train_wvae_light_ar_cond_dof_lowerOnly.py"
    #source_directory = "./train_root_humor.py"
    #source_directory = "./train_root_humor2.py"
    #source_directory = "./train_wvae_light_ar_cond_dof_fullBody_3.py"
    source_directory = "./train_cvae_light_ar_cond_dof3_triton.py"

    
    


    destination_directory = f'./triton/{args.experiment}'
    # Call the function to copy the directory
    copy_directory(source_directory, destination_directory)

    #triton_command = ['python',f'./triton/{args.experiment}/train_cvae_light_ar_cond_dof2.py','--run_name=test', f'--out_name={args.experiment}', ]
    #triton_command = triton_command + args.additional.split()

    triton_command = ['sbatch',f'--output=out/{args.experiment}.out',f'--job-name={args.experiment}', 'new_run.sh', ]
    triton_command.append(args.experiment)
    triton_command.append(args.checkpoint)
    triton_command.append(args.kld_weight)
    triton_command.append(args.latent_size)
    triton_sub = subprocess.Popen(  triton_command)
    triton_sub.wait()
   