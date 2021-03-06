for file in $(find ~/mirplib/t$1/*.json)
do
    ~/master-thesis/target/release/master --problem $file --log info --termination "180 no-improvement" rolling-horizon --checkpoints $1 --checkpoint-termination "7200 timeout 1800 no-improvement no-violation full-empty-valid & & |" --full-penalty-after 180000 --population 3 --children 3 --tournament 2 --step-length 2 --subproblem-size 45 --mutation "std" 2>> log-$(hostname).txt
done
