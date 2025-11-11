% 选择数据集
dataset_choice = 3;

if dataset_choice == 1
    dataset = 'a9a.test';
elseif dataset_choice == 2
    dataset = 'CINA.test';
elseif dataset_choice == 3
    dataset = 'ijcnn1.test';
end
% 读取数据集
[b,A] = libsvmread(dataset);
A = A';
[n,m] = size(A);
% 初始点
x = zeros(n,1);

% 4组优化历史记录各自的索引最大值
index = zeros(4,1);

% 精度 (其中 epsilon_plus 用于高精度近似计算全局最优值)
epsilon = 1e-3;
epsilon_plus = 1e-12; 

% 梯度法最大迭代次数
max_iter = 1000;

% 固定步长 0.05 的梯度法 (精度为 epsilon, 最大迭代 max_iter 次)
fprintf('Constant Stepsize (0.05) Gradient Method:\n');
[history1,index(1)] = Gradient_Iterator(x,A,b,0.05,epsilon,max_iter);
index(1) = index(1) - 1;
fprintf('result = %f\twith %d steps\n',history1(index(1),1),index(1));

% 固定步长 0.10 的梯度法 (精度为 epsilon, 最大迭代 max_iter 次)
fprintf('Constant Stepsize (0.10) Gradient Method:\n');
[history2,index(2)] = Gradient_Iterator(x,A,b,0.10,epsilon,max_iter);
index(2) = index(2) - 1;
fprintf('result = %f\twith %d steps\n',history2(index(2),1),index(2));

% 使用 Armijo 步长准则的梯度法 (精度为 epsilon, 最大迭代 max_iter 次)
fprintf('Backtracking Gradient Method (Armijo Rule):\n');
[history3,index(3)] = Gradient_Iterator(x,A,b,0,epsilon,max_iter);
index(3) = index(3) - 1;
fprintf('result = %f\twith %d steps\n',history3(index(3),1),index(3));

% 精度为 epsilon_plus 的精确牛顿法,以此近似最优值
fprintf('Exact Newton with higher accuracy:\n');
[history4,index(4)] = Newton_Iterator(x,A,b,true,epsilon_plus);
fprintf('result = %f\twith %d steps\n',history4(index(4),1),index(4));
optimal = history4(index(4),1);

%打印算法终止时的目标函数值以及迭代次数
fprintf(['Dataset size:\t (m,n) = (%d,%d)\n' ...
         'Constant Stepsize (0.05) Gradient Method:\t%6.5f\t(%d iteration)\n' ...
         'Constant Stepsize (0.10) Gradient Method:\t%6.5f\t(%d iteration)\n' ...
         'Backtracking Gradient Method (Armijo Rule):\t%6.5f\t(%d iteration)\n' ...
         'optimal:\t%6.5f\n'], ...
         m,n,...
         history1(index(1),1),index(1)-1, ...
         history2(index(2),1),index(2)-1, ...
         history3(index(3),1),index(3)-1, ...
         optimal);

figure
% 绘制梯度范数-迭代数图
subplot(2,1,1);
plot(0:index(1)-1,log10(history1(1:index(1),2)),'b--', ...
     0:index(2)-1,log10(history2(1:index(2),2)),'r-.', ...
     0:index(3)-1,log10(history3(1:index(3),2)),'y-.', ...
     'LineWidth', 2);
% 添加直线 y = log(epsilon)
max_index = max(index);
line(0:max_index-1,log10(epsilon)*ones(max_index,1));
% 添加图例
legend('Constant Stepsize (0.05) Gradient Method', ...
       'Constant Stepsize (0.10) Gradient Method', ...
       'Backtracking Gradient Method (Armijo Rule)');
title('log10(Euclidean norm of gradient)---Number of iterations')
xlabel('Number of iterations') %添加坐标轴标签
ylabel('log10(Euclidean norm of gradient)')

% 绘制残差-迭代数图
subplot(2,1,2);
plot(0:index(1)-1,log10(history1(1:index(1),1)-optimal),'b--', ...
     0:index(2)-1,log10(history2(1:index(2),1)-optimal),'r-.',...
     0:index(3)-1,log10(history3(1:index(3),1)-optimal),'y-.',...
     'LineWidth',2);
% 添加图例
legend('Constant Stepsize (0.05) Gradient Method', ...
       'Constant Stepsize (0.10) Gradient Method', ...
       'Backtracking Gradient Method (Armijo Rule)');
title('log10(Residuals of objective value)---Number of iterations')
xlabel('Number of iterations') % 添加坐标轴标签
ylabel('log10(Residuals of objective value)')

function logistic = Logistic(x,A,b) 
    logistic = 1./(1 + exp(-b.*(A'*x)));
end 

function object = Object(x,A,b)
	[~, m] = size(A);
	lambda = 0.01/m;
	logistic = Logistic(x,A,b);
    object = -(1/m)*(sum(log(logistic))) + lambda * (x'*x);
end

function grad = Gradient(x,A,b) 
	[~, m] = size(A);
	lambda = 0.01/m;
	logistic = Logistic(x,A,b);
	grad = -(1/m)*(A*(b.*(1-logistic))) + 2*lambda*x;
end

function hess = Hessian(x,A,b)
	[n, m] = size(A);
	lambda = 0.01/m;
	logistic = Logistic(x,A,b);	
	vector = (1-logistic).*logistic;
	stack_vector = repmat(vector', n, 1); 
    % 在行维度和列维度上分别重复 vector 的转置 n 次和 1 次, 构造 nxm 的矩阵 stack_vector 
    % 用 (A.*stack_vector)*A' 代替 A*diag(vector)*A', 如果 m>n 的话可以节省空间 
    % 而且将一次矩阵乘法变为矩阵点乘，时间复杂度也降低
    hess = (1/m)*((A.*stack_vector)*A') + 2*lambda*diag(ones(n,1));
end

function stepsize = Armijo(x,A,b,object,grad,direction) 
    alpha = 1e-4;
    beta = 0.5; 
    stepsize = 1; 
    x_new = x + stepsize*direction;
    while Object(x_new,A,b) > (object + alpha * stepsize * (direction' * grad)) 
        stepsize = stepsize * beta; 
    	x_new = x + stepsize*direction;
    end 
end

function CG_tol = Inexact_Rule(ng, CG_tol_rule)
% 'ng' for 'norm of gradient'
% 'CG_tol' for 'tolerance of CG method'
    tol = zeros(3,1);
    tol(1) = min(0.5,ng)*ng;
    tol(2) = min(0.5,sqrt(ng))*ng;
    tol(3) = 0.5*ng;
    CG_tol = tol(CG_tol_rule);
end

function x = CG(A, g, CG_max_iter, CG_tol_rule)
    % CG solve Ax + g = 0
    x = zeros(size(g)); 
    ng = norm(g);
    % Tolerance for convergence
    CG_tol = Inexact_Rule(ng, CG_tol_rule);
    r = g; 
    p = -r;
    
    for iter = 1:CG_max_iter
        % Compute r' * r
        rr = r' * r;
        Ap = A * p;
        stepsize = rr / (p' * Ap);
        % Update solution x
        x = x + stepsize * p;
        % Update residual r
        r = r + stepsize * Ap;
        % Compute the norm of the new residual
        nr = norm(r);
    
        % Check for convergence
        if nr <= CG_tol
            break;
        end

        beta = (nr^2) / rr;
        % Update search direction p
        p = -r + beta * p;
    end
    
    % Print the total number of iterations
    fprintf('Number of CG iteration: %d\t Norm of residual: %f\t', iter, nr);
 
end

function [history,i] = Newton_Iterator(x,A,b,exact_or_not,epsilon,CG_max_iter,CG_tol_rule)
	% 设置默认值
    if nargin < 6
        CG_tol_rule = 1;
        CG_max_iter = 1000;
    end
    if nargin < 5
        epsilon = 1e-6;
    end
    
    % 函数主体
    history = zeros(1001,2);
    i = 1;
    while i <= 1001 
        object = Object(x,A,b);
        grad = Gradient(x,A,b);
        hess = Hessian(x,A,b);
    	
        % 记录优化历史
        history(i,1) = object;
        history(i,2) = norm(grad);

        % 达到精度要求时停止迭代
        if history(i,2) < epsilon
        	break;
        end

        % 精确 Newton 法和非精确 Newton 法的关键区别便是搜索方向的计算
        if exact_or_not
            direction = -hess\grad;
        else 
            direction = CG(hess,grad,CG_max_iter,CG_tol_rule);
        end

        % 根据 Armijo 步长准则选取步长
        stepsize = Armijo(x,A,b,object,grad,direction);

        % 迭代公式
        x = x + stepsize * direction;
        fprintf('%d iteration done!\n',i);
        
        % 更新索引
       	i = i + 1;
    end
end

function [history,i] = Gradient_Iterator(x,A,b,constant_stepsize,epsilon,max_iter)
	% 设置默认值
    if nargin < 6
        max_iter = 1000;
    end
    if nargin < 5
        epsilon = 1e-6;
    end
    if nargin < 4
        constant_stepsize = 0;
    end
    
    % 函数主体
    history = zeros(max_iter+1,2);
    i = 1;
    while i <= max_iter + 1
        object = Object(x,A,b);
        grad = Gradient(x,A,b);
    	
        % 记录优化历史
        history(i,1) = object;
        history(i,2) = norm(grad);

        % 达到精度要求时停止迭代
        if history(i,2) < epsilon
        	break;
        end

		% 梯度法搜索方向
		direction = -grad;

        % 若给定的固定步长为 0, 则根据 Armijo 步长准则选取步长
        if constant_stepsize == 0
            stepsize = Armijo(x,A,b,object,grad,direction);
        else
            % 若给定的固定步长不为 0, 则以其为固定步长
            stepsize = constant_stepsize;
        end

        % 迭代公式
        x = x + stepsize * direction;
        
        % 设置输出语句方便调试
        if mod(i,500)==0
           fprintf('Number of Iteration:\t%d\n',i);
        end
        
        % 更新索引
       	i = i + 1;
    end
end