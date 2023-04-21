function R = postprocessing(R,l,w,wb,dt)
% function to compute the space occupied by the car from the reachable set

    R_ = R;

    % shape of the car
    car = zonotope(interval([-(l-wb)/2;-w/2],[wb + (l-wb)/2;w/2]));
    
    % define function to compute occupancy set
    f = @(x,p) [x(1) + cos(x(4))*p(1) - sin(x(4))*p(2);
                x(2) + cos(x(4))*p(2) + sin(x(4))*p(1)];

    % zonotope order
    order = 3;

    % get transformation function in symbolic form
    x = sym('x',[dim(R{1}.set),1]);
    p = sym('p',[dim(car),1]);

    fsym = f(x,p);

    % determine states that appear in the function
    tmp = (1:length(x));
    ind = tmp(ismember(x,symvar(fsym)));
    
    % compute derivatives
    [fun,Afun,Qfun,Tfun] = computeDerivatives(fsym,[x(ind);p]);    
    
    % loop over all reachable sets
    for i = 1:length(R)
    
        % reduce the zonotope order
        set = reduce(polyZonotope(R{i}.set),'girard',order);
        
        % define initial set
        X = cartProd(project(set,ind),car);
        
        % evaluate derivatives at linearization point
        p = center(X);

        [f,A,Q,T] = evalDerivatives(X,p,fun,Afun,Qfun,Tfun);

        % compute Largrange remainder
        rem = lagrangeRemainder(X,p,T);

        % compute over-approximating zonotope
        res = f + exactPlus(A * (X + (-p)), 0.5*quadMap((X + (-p)),Q)) + rem;

        % convert to polygon object
        R{i}.set = res;
    end
    
    % unite occopancy sets to speed-up collision checking
    N = round(supremum(R{end}.time)/dt);
    t = linspace(infimum(R{1}.time),supremum(R{end}.time),N+1);
    O = cell(N,1);
    counter = 2;
    
    for i = 1:N

       start = counter - 1;

       while counter <= length(R) && infimum(R{counter}.time) < t(i+1)-eps
          counter = counter + 1;
       end
       
       p1 = center(R_{start}.set);
       p2 = center(R_{counter-1}.set);
       diff = p1 - p2;

       C = [cos(p1(4)) sin(p1(4))];
       C = [C; -C(end,:)];
       C = [C; -sin(p1(4)) cos(p1(4))];
       C = [C; -C(end,:)];
       C = [C; cos(p2(4)) sin(p2(4))];
       C = [C; -C(end,:)];
       C = [C; -sin(p2(4)) cos(p2(4))];
       C = [C; -C(end,:)];
       C = [C; diff(1:2)'];
       C = [C; -C(end,:)];
       C = [C; [diff(2) -diff(1)]];
       C = [C; -C(end,:)];

       d = -inf * ones(size(C,1), 1);

       for j = start:counter-1
           for k = 1:size(C,1)
               d(k) = max(d(k), supportFunc(R{j}.set, C(k,:)));
           end
       end

       V = vertices(mptPolytope(C,d));
       ind = convhull(V');
       V = V(:,ind);

       w = warning();
       warning('off');

       O{i}.set = polygon(V(1,:),V(2,:));

       warning(w);

       O{i}.time = interval(t(i),t(i+1));
    end

%     % visualization
%     figure; hold on;
% 
%     for i = 1:length(R_)
%         for j = 1:10
%             p = randPoint(R_{i}.set);
%             car_ = p(1:2) + [cos(p(4)) -sin(p(4)); sin(p(4)) cos(p(4))] * car;
%             plot(car_,[1,2],'b');
%         end
%     end
% 
%     for i = 1:length(O)
%         plot(O{i}.set,[1,2],'r');
%     end
end


% Auxiliary Functions -----------------------------------------------------

function [fun,Afun,Qfun,Tfun] = computeDerivatives(f,x)
% compute the symbolic derivatives of the function

    % function handle for the nonlinear function
    fun =  matlabFunction(f,'Vars',{x});
    
    A = jacobian(f,x);
    Afun =  matlabFunction(A,'Vars',{x});
    
    Qfun = cell(length(f),1);
    for i = 1:length(f)
       temp = hessian(f(i),x); 
       Qfun{i} =  matlabFunction(temp,'Vars',{x});
    end
    
    Tfun = cell(size(A));
    for i = 1:size(A,1)
        for j = 1:size(A,2)
            temp = hessian(A(i,j),x);
            Tfun{i,j} = matlabFunction(temp,'Vars',{x});
        end
    end
end

function [f,A,Q,T] = evalDerivatives(X,p,fun,Afun,Qfun,Tfun)
% evaluate the derivatives at the linearization point

    % interval encluore of the set
    int = interval(X);
    
    f = fun(p);
    A = Afun(p);
    
    Q = cell(length(f),1);
    for i = 1:length(f)
       funHan = Qfun{i};
       Q{i} = funHan(p);
    end
    
    T = cell(size(A));
    for i = 1:size(A,1)
        for j = 1:size(A,2)
            funHan = Tfun{i,j};
            T{i,j} = funHan(int);
        end
    end
end

function rem = lagrangeRemainder(X,p,T)
% comptute the lagrange remainder of the Taylor series

    % interval enclousre of the shifted initial set
    int = interval(X) - p;

    % Lagrange remainder term
    rem = interval(zeros(size(T,1),1));
    
    for i = 1:size(T,1)
        for j = 1:size(T,2)
            rem(i) = rem(i) + int(j) * transpose(int) * T{i,j} * int;
        end
    end
    
    % convert to zonotope
    rem = zonotope(1/6*rem);
end