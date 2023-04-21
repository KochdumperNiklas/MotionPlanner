function f = vehicle(x,u,w)
% differential equation desribing the behaviour of the car

    wb = 2.3;

    f(1,1) = x(3) * cos(x(4));
    f(2,1) = x(3) * sin(x(4));
    f(3,1) = u(1) + w(1);
    f(4,1) = x(3) * tan(u(2)) / wb + w(2);