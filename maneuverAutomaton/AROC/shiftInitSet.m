function Rinit = shiftInitSet(Rinit,xf)
% shift initial set to the final state of previous maneuver

    % shift the set to the origin of the position space
    c = center(Rinit);
    c(3) = 0;
    
    Rinit = zonotope(Rinit + (-c));
    
    % rotate the set by the orientation of the previous maneuver
    phi = xf(4);
    T = blkdiag([cos(phi) -sin(phi); sin(phi) cos(phi)],eye(2));
    Tinv = blkdiag([cos(c(2)) -sin(c(2)); sin(c(2)) cos(c(2))],eye(2)');
    
    Rinit = T * Tinv * Rinit;
    
    % shift the set to the final state of the previous maneuver
    xf(3) = 0;
    Rinit = Rinit + xf;
end
