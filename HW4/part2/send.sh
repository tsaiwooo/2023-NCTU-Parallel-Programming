hosts_file="host.txt"
file_to_copy="matmul"
destination_path="/home/312551129/"

while IFS= read -r host; do
    echo "Copying $file_to_copy to $host"
    scp "$file_to_copy" "$host":"$destination_path"
done < "$hosts_file"