
export i=0;
export pids="";
for file in $(find ~/mirplib/t$1/*.json)
do
    echo "i=$i: Starting run on $file"
    ~/master-thesis/target/release/master --problem $file --log info --termination "180 no-improvement" rolling-horizon --checkpoints $1 --checkpoint-termination "7200 timeout 1800 no-improvement no-violation full-empty-valid & & |" --full-penalty-after 180000 --population 3 --children 3 --tournament 2 --step-length 2 --subproblem-size 45 --mutation "lite" --threads 1 2>> log-$(hostname)-$i.txt &
    pids="$pids $!"
    i=$((i + 1))

    # Wait for all of the runs to complete before starting a new set.
    # Could've done this better
    if [ $i -eq $2 ]; then
        wait $pids
        i=$((0))
        pids=""
    fi
done

wait $pids