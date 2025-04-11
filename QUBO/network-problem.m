N = 20; % Number of devices;
k = 4; % Number of routers;
rng default % For reproducibility
cX = 1.5*rand(N,1);
cY = rand(N,1);

%cX = [1,1/2,-1/2,-1,-1/2,1/2];
%cY = [0,sqrt(3)/2,sqrt(3)/2,0,-sqrt(3)/2,-sqrt(3)/2];

[X,Y] = meshgrid(1:N);
dist = hypot(cY(X) - cY(Y),cX(X) - cX(Y));

disp(dist)

Q1 = dist2qubo(dist,N,k);
result = solve(Q1);

binx = result.BestX;
binx = reshape(binx,N,N+1)';
disp(binx)
disp(result.BestFunctionValue)

hold on 
for i = 1:N
    for j = 1:N
        if binx(i,j) > 0
            plot([cX(i),cX(j)],[cY(i),cY(j)],'-o','Color','r');
        end
    end
end

for j = 1:N
    if binx(N+1,j) > 0
        plot(cX(j), cY(j), 'bo', 'MarkerSize',10, 'MarkerFaceColor', 'b'); 
    end;
end;

hold off 

function R = dist2qubo(dist,N,k)

% N = number of devices
% k = number of routers
% dist = N*N matrix representing distance between each pair of device

A = eye(N+1);
B = ones(N);
Q = kron(A,B);
for i = N*N+1:N*N+N
    for j = 1:N
        row = (j-1)*N + mod(i-1,N*N) + 1;
        Q(row,i) = -1;
    end
end

M = max(max(dist));
M = M*M*N^2;

d = N + k*k;

C0 = zeros(1,N*N+N);
C1 = zeros(1,N*N+N);
for i = 1:N*N
    C0(1,i) = dist(floor((i-1)/N)+1,mod(i-1,N)+1);
    C1(1,i) = -1;
end
for i = N*N+1:N*N+N
    C1(1,i) = -2*k;
end

Q = (Q + Q.')/2;

R = qubo(Q*M,C0 + C1*M,d*M);

end

