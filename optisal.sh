cd ~/master-thesis && cargo build --release;

export i=0;
export pids="";
for file in $(find ~/master-playground/mirplib-rs/t$1/*.json)
do
    echo "i=$i: Starting run on $file"
    ~/master-thesis/target/release/master --problem $file --log info --termination "120 no-improvement" rolling-horizon --checkpoint-termination "10800 timeout 1800 no-improvement no-violation full-empty-valid & & |" --threads 1 --full-penalty-after 3600000 --population 3 --children 3 --tournament 2 --subproblem-size 45 --step-length 2 --checkpoints $1 --mutation "std" 2>> log-$(hostname)-$i.txt &
    pids="$pids $!"
    i=$((i + 1))

    if [ $i -eq $2 ]; then
        wait $pids
        i=$((0))
        pids=""
    fi
done

wait $pids