while true
do
	timeout -k 10 3600 /home/niklaskochdumper/MATLAB/R2022a/bin/matlab -r createManeuverAutomaton;
	sleep 2
done
