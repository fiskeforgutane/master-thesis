
for difficulty in $1
do
    for time in 145 235 290
    do
        for file in $(find ~/mowilib/real/t$time/$difficulty/*.json)
        do
            ~/master-thesis/target/release/master --problem $file --log info --termination "1800 no-improvement no-violation & 10800 timeout |" rolling-horizon --full-penalty-after 360000 --population 3 --children 3 --tournament 2 --step-length 2 --subproblem-size $time --mutation "lite" --travel-at-cap 0 --travel-empty 0 2>> log-$(hostname).txt
        done
    done
done
