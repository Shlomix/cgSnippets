#!/bin/bash

# Function to download .bin files
download_bin_files() {
    local model_name=$1
    local bin_files=$(find . -name "*.bin")
    
    for bin_file in $bin_files; do
        # Construct the URL for the .bin file
        bin_file_name=$(basename "$bin_file")
        url="https://huggingface.co/${model_name}/resolve/main/${bin_file_name}"
        
        # Download the .bin file
        echo "Downloading ${bin_file} from ${url}..."
        curl -L -o "$bin_file" "$url"
        
        if [[ $? -ne 0 ]]; then
            echo "Failed to download ${bin_file}"
        else
            echo "Successfully downloaded ${bin_file}"
        fi
    done
}

# Main script execution
if [[ -z "$1" ]]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

model_name=$1

# Extract the directory name from the model name
dir_name=$(basename "${model_name}")

# Command to clone the repo and download the bin files
cmd=$(cat <<EOF
#!/bin/bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/${model_name}
if [[ $? -ne 0 ]]; then
    echo "Failed to clone repository https://huggingface.co/${model_name}"
    exit 1
fi
cd "${dir_name}" || exit 1
$(declare -f download_bin_files)
download_bin_files "${model_name}"
EOF
)

# Start a new screen session and run the command
screen -S "${model_name}" -d -m bash -c "$cmd"

# Attach to the screen session
screen -r "${model_name}"
