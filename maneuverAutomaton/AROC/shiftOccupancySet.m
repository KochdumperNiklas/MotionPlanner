function R = shiftOccupancySet_car(R,xf,time)
% shift occupancy set to final state of previous maneuver

    % compute tranformation matrix and offset
    phi = xf(4);
    T = [cos(phi) -sin(phi); sin(phi) cos(phi)];
    o = xf(1:2);
    
    % loop over all reachable sets
    for i = 1:length(R)
        % transform the set
        R{i}.set = T*R{i}.set + o;
        R{i}.time = R{i}.time + time;
    end
end
