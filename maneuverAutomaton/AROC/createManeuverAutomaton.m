warning('off');

% vehicle parameter
s_max = 1;                          % maximum steering angle [rad]
a_max = 9;                          % maximum acceleration [m/s^2]
wb = 2.3;                           % length of wheelbase of the car [m]
l = 4.3;                            % length of the car [m]
w = 1.7;                            % width of the car [m]

% time step size (for occupancy set computation)
dt = 0.1;

% controller settings
Opts = [];

Opts.N = 1;                         % number of time steps
Opts.Ninter = 10;                   % number of intermediate time steps
Opts.reachSteps = 10;                % number of reachability steps
Opts.Q = diag([1 5 1 10]);          % state weigthing matrix

Opts.cora.tensorOrder = 3;
Opts.cora.lagrangeRem.simplify = 'optimize'; 

% postprocessing function (used to compute the occupancy set)
Post = @(x) postprocessing(x,l,w,wb,dt);

% set of admissible control inputs
width = [a_max;s_max];
U_max = interval(-width,width);

% desired set of control inputs
width = [8;0.4];
U_des = interval(-width,width);

% set of uncertain disturbances
width = [0;0];
Param.W = interval(-width,width);

% initial set of states
x0 = [0;0;0;0];
width = [0.1; 0.1; 0.2; 0.02];
R0 = interval(x0-width,x0+width);

% path where the motion primitives are stored
[path,~,~] = fileparts(which(mfilename));
pathSuccess = fullfile(path,'primitives','success');
pathFail = fullfile(path,'primitives','fail');

if ~isfolder(pathSuccess)
    mkdir(pathSuccess);
end

if ~isfolder(pathFail)
    mkdir(pathFail);
end

% acceleration
accelerations = [0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 2, -2, 4, -4, 8, -8];

% desired final orientation
orientation = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1];

% velocity range
v_start = 0;
v_end = 30;
v_diff = 0.4;

v = v_start:v_diff:v_end;

% final time
tFinal = 1;

% loop over all initial velocities
parfor i = 1:length(v)

    v_init = v(i);
    Param_ = Param;
    Opts_ = Opts;

    % loop over all accelerations
    for j = 1:length(accelerations)

        acc = accelerations(j);

        % modify acceleration and final time to ensure that the motion 
        % primitive can be connected to others
        if v_start <= v_init + acc*tFinal && v_init + acc*tFinal <= v_end
            tFinal_ = tFinal;
        elseif v_init + acc*tFinal < v_start
            tFinal_ = round(((v_start - v_init)/acc)/dt) * dt;
            acc = (v_start - v_init)/tFinal_;
        else
            tFinal_ = round(((v_start - v_end) / acc)/dt) * dt;
            acc = (v_start - v_end) / tFinal_;
        end

        if tFinal_ > 0

            % loop over all final orientations
            for k = 1:length(orientation)

                o = orientation(k);
                name = ['primitive_',num2str(i),'_',num2str(j),'_',num2str(k),'.mat'];
                nameMirror = ['primitive_',num2str(i),'_',num2str(j),'_',num2str(k + length(orientation)),'.mat'];
                fileSuccess = fullfile(pathSuccess,name);
                fileFail = fullfile(pathFail,name);
                fileMirror = fullfile(pathSuccess,nameMirror);

                if ~isfile(fileFail) && ~isfile(fileSuccess) && (o == 0 || abs(v_init * tFinal_ + 0.5*acc * tFinal_^2) > 0)

                    % compute the required steering angle to achieve the desired final orientation
                    if o == 0
                        steer = 0;
                    else
                        steer = atan(wb * o / (v_init * tFinal_ + 0.5*acc * tFinal_^2));
                    end

                    if abs(steer) < s_max

                        % simulate the system to get reference trajectory
                        x0 = [0;0;v_init;0];
                        u = [acc; steer];
                        tspan = 0:tFinal_/(Opts_.N*Opts_.Ninter):tFinal_;
                        fun = @(t,x) vehicle(x,u,zeros(2,1));

                        [t,x] = ode45(fun,tspan,x0);

                        % update parameters
                        Param_.tFinal = tFinal_;
                        Param_.U = U_max & (u + U_des);
                        Param_.R0 = R0 + [0;0;v_init;0];
                        Param_.xf = x(end,:)';
                        Opts_.refTraj.x = x';
                        Opts_.refTraj.u = u*ones(1,size(x,1)-1);

                        % optimize weighting matrix
                        Opts_ = optimizeWeightingMatrix(Param_,Opts_);

                        % synthesise controller for the motion primitive
                        [obj,res] = generatorSpaceControl('vehicle',Param_,Opts_,Post);

                        % check if the motion primitive can be connected
                        xf = center(obj.Rfin);
                        set = zonotope(Param_.R0 + (-center(Param_.R0)));
                        phi = xf(4);
                        set = blkdiag([cos(phi) -sin(phi); sin(phi) cos(phi)],eye(2)) * set + xf;

                        if contains(set,obj.Rfin)
                            disp([name,':     success (acc=',num2str(acc),',steer=',num2str(steer),')']);
                            parsave(fileSuccess,obj);
                            if abs(steer) > 0
                                objMirror = mirror(obj,[2,4],2,2);
                                parsave(fileMirror,objMirror);
                            end
                        else
                            disp([name,':     failed (acc=',num2str(acc),',steer=',num2str(steer),')']);
                            parsave(fileFail,obj);
                        end
                    end
                end
            end
        end
    end
end

% load files and create maneuver automaton
files = dir(pathSuccess);
primitives = {};

for k = 1:length(files)
    if ~ismember(files(k).name, {'.', '..'})
        load(fullfile(pathSuccess,files(k).name));
        primitives{end+1} = obj;
    end
end

MA = maneuverAutomaton(primitives,@shiftInitSet,@shiftOccupancySet);

save(fullfile(path),'automaton.mat'), 'MA');

% export maneuver automaton to .xml-file
file = fullfile(path,'automaton.zip');
states = {'x','y','velocity','orientation'};
inputs = {'acceleration','steer'};

if ~isfile(file)
    exportXML(MA,file,states,inputs);
end


% Auxiliary Functions -----------------------------------------------------

function parsave(fname, obj)
  save(fname, 'obj');
end

function Opts = optimizeWeightingMatrix(Param,Opts)

    for i = 1:10

        % controller synthesis
        [obj,~] = generatorSpaceControl('vehicle',Param,Opts);
        
        % check if final set is contained in shifted initial set
        xf = center(obj.Rfin);
        set = zonotope(Param.R0 + (-center(Param.R0)));
        phi = xf(4);
        set = blkdiag([cos(phi) -sin(phi); sin(phi) cos(phi)],eye(2)) * set + xf;
    
        if contains(set,obj.Rfin)
            break;
        end
        
        % update weighting matrix
        scale = rad(interval(obj.Rfin))./rad(interval(Param.R0));
        scale = scale./scale(1);
        Opts.Q = diag(scale);

    end
end
