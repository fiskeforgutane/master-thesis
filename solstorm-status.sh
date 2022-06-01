for file in $(find log-compute*.txt)
do
        problem=$(grep "problem: " $file | tail -n 1)
        timesteps=$(grep "timestep" $file | tail -n 1)
        termination=$(grep "termination" $file | tail -n 1)
        fitness=$(grep "F = " $file | tail -n 1)
        subproblem=$(grep "subproblem" $file | tail -n 1)
        echo "$file"
        echo "$problem"
        echo "$timesteps"
        echo "$termination"
        echo "$subproblem"
        echo "$fitness"
        echo ""
done