modes=(2 3) 
for d in "${modes[@]}" 
do 
    total=100 # adjust based on your needs 
    # breakdown 100 -> 80 zoo 20 teacher-school 20 movies (100 for reverse mode as well) 

    lengths=(zero_context 8k 16k 32k) # adjust if needed 

    for length in "${lengths[@]}" 
    do 
        python datagenerationworker.py --numprocs 8 --opmax 30 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 20 19 
        python datagenerationworker.py --numprocs 8 --opmax 25 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 16 17 18 
        python datagenerationworker.py --numprocs 8 --opmax 20 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 12 13 14 15 
        python datagenerationworker.py --numprocs 8 --opmax 15 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 10 11 
        python datagenerationworker.py --numprocs 8 --opmax 10 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 7 8 9 
        python datagenerationworker.py --numprocs 16 --opmax 6 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 5 6 
        python datagenerationworker.py --numprocs 16 --opmax 4 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 4 
        python datagenerationworker.py --numprocs 16 --opmax 4 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 3 
        python datagenerationworker.py --numprocs 16 --opmax 3 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 2 
    done 
done 
