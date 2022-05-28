cd ~/master-thesis && cargo build --release;

export i=0;
export pids="";
for file in $(find ~/master-playground/factory_locations/N40_V7*.json)
do
    echo "i=$i: Starting run on $file" >> log-$(hostname)-$i.txt
    ~/master-thesis/target/release/master --problem $file --log info --termination "120 no-improvement" rolling-horizon --checkpoint-termination "3600 timeout 1800 no-improvement no-violation & |" --threads 8 --full-penalty-after 36000 --population 3 --children 3 --tournament 2 --subproblem-size 145 --step-length 2 --checkpoints $1 --mutation "std" --travel-at-cap 0 --travel-empty 0 2>> log-$(hostname)-$i.txt &
    pids="$pids $!"
    i=$((i + 1))

    if [ $i -eq $2 ]; then
        wait $pids
        i=$((0))
        pids=""
    fi
done

wait $pids