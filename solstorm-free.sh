for file in $(find ~/mirplib/t120/*.json)
do
    ~/master-thesis/target/release/master --problem $file --log info --termination "300 no-improvement" rolling-horizon --checkpoints 45 --checkpoints 60 --checkpoints 90 --checkpoints 120 --checkpoint-termination "1800 no-improvement" --full-penalty-after 360000 --population 3 --children 3 --tournament 2 --step-length 2 --subproblem-size 30 --mutation "lite" --travel-at-cap 0 --travel-empty 0 2>> log-$(hostname).txt
done
