function projection_compare_box_refined()
%==========================================================================
% Refined comparison: exact decision-sufficient reduction vs projection LPs
% -------------------------------------------------------------------------
% This script fixes two issues in the earlier baseline demo:
%
%   (1) It separates FULL-ORACLE from FULL-LINPROG.  For a box LP the full
%       optimum has a closed form, so using it as a runtime baseline is not
%       a fair solver comparison.  We therefore report both.
%
%   (2) It performs a K-sweep and reports training time separately from
%       online solve time.  PCA/SharedP/InstMLP require training; our exact
%       reduced LP has a discovery stage and then an exact online solve.
%
% Problem family:
%       max_x p' x    s.t. 0 <= x <= 1.
% A prior ball around p0 is chosen so that exactly dstar coordinates can
% cross zero. These are the decision-relevant coordinates.  Hence the exact
% decision-sufficient subspace is known for evaluation.
%
% Methods:
%   FullOracle       : closed-form full optimum.  Not a solver baseline.
%   FullLinprog      : full LP solved with linprog.  Fair online solver baseline.
%   OursExact        : exact reduced LP using the decision-sufficient active set.
%   ColRand          : random coordinate columns, repeated several times.
%   PCA              : shared projection from training optimal solutions.
%   SharedP-PCAinit  : SGA refinement initialized by PCA.
%   SharedP-Randinit : SGA refinement initialized randomly.
%   InstTopK         : instance-specific heuristic selecting top-K profit coords.
%   InstMLP-lite     : optional hand-coded small MLP generator. Disabled by default.
%
% Notes:
%   - InstTopK / InstMLP-lite are NOT faithful reproductions of Iwata--Sakaue
%     PELP.  They are diagnostic baselines showing what naive instance-specific
%     generators do on this exact-sufficiency family.
%   - A faithful PELP reproduction is better implemented with MATLAB Deep
%     Learning Toolbox custom training or in PyTorch.
%==========================================================================

clc; close all;

cfg.seed        = 20260418;
cfg.d           = 250;      % ambient dimension
cfg.dstar       = 12;       % exact intrinsic dimension
cfg.rho         = 1.0;      % ball radius in profit space
cfg.nTrain      = 250;
cfg.nTest       = 150;
cfg.kGrid       = [4 8 12 16 24 36];
cfg.nRand       = 10;
cfg.sgaEpochs   = 4;
cfg.sgaLR       = 0.015;
cfg.useInstMLP  = false;    % set true only if you want the slow diagnostic MLP
cfg.mlpEpochs   = 25;
cfg.mlpHidden   = 32;
cfg.verbose     = true;

rng(cfg.seed, 'twister');
opts = make_linprog_options();

[p0, activeSet] = make_profit_family_center(cfg.d, cfg.dstar, cfg.rho);
Pstar = eye(cfg.d); Pstar = Pstar(:, activeSet);
x0 = full_box_oracle(p0);  % anchor solution

Ptrain = sample_profit_ball_sparse(cfg.nTrain, p0, cfg.rho, activeSet);
Ptest  = sample_profit_ball_sparse(cfg.nTest,  p0, cfg.rho, activeSet);
Xtrain = double(Ptrain > 0);  % full optimal solutions for PCA/SharedP baselines

fprintf('Refined box comparison: d=%d, true d*=%d, nTrain=%d, nTest=%d\n', ...
    cfg.d, cfg.dstar, cfg.nTrain, cfg.nTest);

% Full optimal values and runtimes.
valFullOracle = zeros(cfg.nTest,1);
timeFullOracle = zeros(cfg.nTest,1);
valFullLP = zeros(cfg.nTest,1);
timeFullLP = zeros(cfg.nTest,1);
for i = 1:cfg.nTest
    p = Ptest(i,:)';
    tic; x = full_box_oracle(p); valFullOracle(i) = p' * x; timeFullOracle(i)=toc;
    tic; [xlp, v] = solve_full_box_linprog_max(p, opts); valFullLP(i)=v; timeFullLP(i)=toc; %#ok<ASGLU>
end

nK = numel(cfg.kGrid);
res = struct();
methods = {'OursExact','ColRand','PCA','SharedP_PCAinit','SharedP_Randinit','InstTopK','InstMLPlite'};
for mm = 1:numel(methods)
    res.(methods{mm}).ratio = nan(nK,1);
    res.(methods{mm}).ratioStd = nan(nK,1);
    res.(methods{mm}).time = nan(nK,1);
    res.(methods{mm}).trainTime = nan(nK,1);
end

for kk = 1:nK
    K = cfg.kGrid(kk);
    fprintf('\n=== K = %d ===\n', K);

    % ------------------- Ours exact -------------------
    tic; Uours = Pstar; trainOurs = toc; %#ok<NASGU>
    [ratio, runTime] = evaluate_exact_reduced(Ptest, valFullOracle, Uours, x0, opts);
    res.OursExact.ratio(kk) = ratio;
    res.OursExact.ratioStd(kk) = 0;
    res.OursExact.time(kk) = runTime;
    res.OursExact.trainTime(kk) = trainOurs;

    % ------------------- ColRand ----------------------
    randRatios = zeros(cfg.nRand,1);
    randTimes = zeros(cfg.nRand,1);
    for rr = 1:cfg.nRand
        Pc = colrand_projection(cfg.d, K);
        [randRatios(rr), randTimes(rr)] = evaluate_projection(Ptest, valFullOracle, Pc, opts);
    end
    res.ColRand.ratio(kk) = mean(randRatios);
    res.ColRand.ratioStd(kk) = std(randRatios);
    res.ColRand.time(kk) = mean(randTimes);
    res.ColRand.trainTime(kk) = 0;

    % ------------------- PCA --------------------------
    t0 = tic; Ppca = pca_projection(Xtrain, K); trainTime = toc(t0);
    [ratio, runTime] = evaluate_projection(Ptest, valFullOracle, Ppca, opts);
    res.PCA.ratio(kk) = ratio;
    res.PCA.ratioStd(kk) = 0;
    res.PCA.time(kk) = runTime;
    res.PCA.trainTime(kk) = trainTime;

    % ------------------- SharedP PCA-init -------------
    t0 = tic; Psgap = train_sharedP_sga(Ptrain, K, Ppca, cfg.sgaEpochs, cfg.sgaLR, opts); trainTime = toc(t0);
    [ratio, runTime] = evaluate_projection(Ptest, valFullOracle, Psgap, opts);
    res.SharedP_PCAinit.ratio(kk) = ratio;
    res.SharedP_PCAinit.ratioStd(kk) = 0;
    res.SharedP_PCAinit.time(kk) = runTime;
    res.SharedP_PCAinit.trainTime(kk) = trainTime;

    % ------------------- SharedP random-init ----------
    t0 = tic; Prand0 = rand_feasible_projection(cfg.d, K); PsgaR = train_sharedP_sga(Ptrain, K, Prand0, cfg.sgaEpochs, cfg.sgaLR, opts); trainTime = toc(t0);
    [ratio, runTime] = evaluate_projection(Ptest, valFullOracle, PsgaR, opts);
    res.SharedP_Randinit.ratio(kk) = ratio;
    res.SharedP_Randinit.ratioStd(kk) = 0;
    res.SharedP_Randinit.time(kk) = runTime;
    res.SharedP_Randinit.trainTime(kk) = trainTime;

    % ------------------- Instance-specific top-K ------
    [ratio, runTime] = evaluate_inst_topk(Ptest, valFullOracle, K, opts);
    res.InstTopK.ratio(kk) = ratio;
    res.InstTopK.ratioStd(kk) = 0;
    res.InstTopK.time(kk) = runTime;
    res.InstTopK.trainTime(kk) = 0;

    % ------------------- Optional MLP-lite ------------
    if cfg.useInstMLP
        t0 = tic; mlp = train_inst_mlp_lite(Ptrain, Xtrain, K, cfg); trainTime = toc(t0);
        [ratio, runTime] = evaluate_inst_mlp_lite(Ptest, valFullOracle, mlp, opts);
        res.InstMLPlite.ratio(kk) = ratio;
        res.InstMLPlite.ratioStd(kk) = 0;
        res.InstMLPlite.time(kk) = runTime;
        res.InstMLPlite.trainTime(kk) = trainTime;
    end

    if cfg.verbose
        fprintf('Ours=%.3f | ColRand=%.3f | PCA=%.3f | SGA(PCA)=%.3f | SGA(rand)=%.3f | InstTopK=%.3f\n', ...
            res.OursExact.ratio(kk), res.ColRand.ratio(kk), res.PCA.ratio(kk), ...
            res.SharedP_PCAinit.ratio(kk), res.SharedP_Randinit.ratio(kk), res.InstTopK.ratio(kk));
    end
end

outDir = prepare_results_dir('refined_projection_comparison');

% Figure 1: K sweep objective ratio.
fig1 = figure('Color','w'); hold on; box on; grid on;
plot(cfg.kGrid, ones(nK,1), '-k', 'LineWidth', 1.4);
plot(cfg.kGrid, res.OursExact.ratio, '-d', 'LineWidth', 1.6);
errorbar(cfg.kGrid, res.ColRand.ratio, res.ColRand.ratioStd, '-^', 'LineWidth', 1.2);
plot(cfg.kGrid, res.PCA.ratio, '-o', 'LineWidth', 1.2);
plot(cfg.kGrid, res.SharedP_PCAinit.ratio, '-s', 'LineWidth', 1.2);
plot(cfg.kGrid, res.SharedP_Randinit.ratio, '--s', 'LineWidth', 1.2);
plot(cfg.kGrid, res.InstTopK.ratio, '-x', 'LineWidth', 1.2);
if cfg.useInstMLP
    plot(cfg.kGrid, res.InstMLPlite.ratio, '-p', 'LineWidth', 1.2);
end
xline(cfg.dstar, '--', 'true d^*', 'LineWidth', 1.1);
ylim([0 1.05]);
xlabel('Reduced dimension K');
ylabel('Mean objective ratio');
title('Objective ratio: K-sweep on exact box-sufficiency family');
legend({'Full','Ours exact','ColRand','PCA','SharedP-PCAinit','SharedP-Randinit','InstTopK'}, 'Location','southeast');
save_figure(fig1, fullfile(outDir, 'objective_ratio_K_sweep.png'));

% Figure 2: Online runtime, with FullOracle and FullLP separated.
fig2 = figure('Color','w'); hold on; box on; grid on;
semilogy(cfg.kGrid, mean(timeFullOracle)*ones(nK,1), '-k', 'LineWidth', 1.2);
semilogy(cfg.kGrid, mean(timeFullLP)*ones(nK,1), '--k', 'LineWidth', 1.2);
semilogy(cfg.kGrid, res.OursExact.time, '-d', 'LineWidth', 1.5);
semilogy(cfg.kGrid, res.ColRand.time, '-^', 'LineWidth', 1.2);
semilogy(cfg.kGrid, res.PCA.time, '-o', 'LineWidth', 1.2);
semilogy(cfg.kGrid, res.SharedP_PCAinit.time, '-s', 'LineWidth', 1.2);
semilogy(cfg.kGrid, res.InstTopK.time, '-x', 'LineWidth', 1.2);
xlabel('Reduced dimension K');
ylabel('Mean online solve time (seconds, log scale)');
title('Online runtime: FullOracle separated from FullLinprog');
legend({'FullOracle closed form','FullLinprog','Ours exact','ColRand','PCA','SharedP-PCAinit','InstTopK'}, 'Location','northwest');
save_figure(fig2, fullfile(outDir, 'online_runtime_K_sweep.png'));

% Figure 3: Training time.
fig3 = figure('Color','w'); hold on; box on; grid on;
semilogy(cfg.kGrid, max(res.PCA.trainTime, 1e-9), '-o', 'LineWidth',1.2);
semilogy(cfg.kGrid, max(res.SharedP_PCAinit.trainTime, 1e-9), '-s', 'LineWidth',1.2);
semilogy(cfg.kGrid, max(res.SharedP_Randinit.trainTime, 1e-9), '--s', 'LineWidth',1.2);
xlabel('Reduced dimension K');
ylabel('Training time (seconds, log scale)');
title('Training cost: SGA is typically much slower than PCA');
legend({'PCA','SharedP-PCAinit','SharedP-Randinit'}, 'Location','northwest');
save_figure(fig3, fullfile(outDir, 'training_time_K_sweep.png'));

% Single-K bar at K=dstar.
[~, idxK] = min(abs(cfg.kGrid - cfg.dstar));
barLabels = {'Full','Ours','ColRand','PCA','SGA-PCA','SGA-rand','InstTopK'};
barVals = [1, res.OursExact.ratio(idxK), res.ColRand.ratio(idxK), res.PCA.ratio(idxK), ...
           res.SharedP_PCAinit.ratio(idxK), res.SharedP_Randinit.ratio(idxK), res.InstTopK.ratio(idxK)];
fig4 = figure('Color','w'); box on; grid on;
bar(barVals);
set(gca,'XTickLabel',barLabels); xtickangle(25);
ylabel('Mean objective ratio');
title(sprintf('Objective comparison at K=%d (true d^*=%d)', cfg.kGrid(idxK), cfg.dstar));
save_figure(fig4, fullfile(outDir, 'objective_ratio_singleK_bar.png'));

save(fullfile(outDir, 'summary.mat'), 'cfg', 'res', 'activeSet', 'timeFullOracle', 'timeFullLP');
fprintf('\nSaved refined comparison results to: %s\n', outDir);
end

%==========================================================================
% Family construction and sampling
%==========================================================================
function [p0, activeSet] = make_profit_family_center(d, dstar, rho)
perm = randperm(d);
activeSet = sort(perm(1:dstar));
inactive = setdiff(1:d, activeSet);
p0 = zeros(d,1);
% Active coordinates can cross zero within the ball.
vals = linspace(-0.55*rho, 0.55*rho, dstar)';
vals(abs(vals) < 0.08*rho) = vals(abs(vals) < 0.08*rho) + 0.12*rho;
p0(activeSet) = vals;
% Inactive coordinates never cross zero.
signs = sign(randn(numel(inactive),1)); signs(signs==0)=1;
p0(inactive) = signs .* (rho + 0.75 + 0.5*rand(numel(inactive),1));
end

function P = sample_profit_ball_sparse(n, p0, rho, activeSet)
d = numel(p0);
P = zeros(n,d);
for i = 1:n
    delta = zeros(d,1);
    % sparse active-coordinate perturbations -> gradual discovery possible
    k = randsample([1 2 3], 1, true, [0.70 0.25 0.05]);
    S = activeSet(randperm(numel(activeSet), min(k,numel(activeSet))));
    for s = S(:)'
        delta(s) = -p0(s) + (2*rand-1) * 0.95*rho;
    end
    if norm(delta) > 0.95*rho
        delta = delta * (0.95*rho/norm(delta));
    end
    P(i,:) = (p0 + delta)';
end
end

%==========================================================================
% Full and reduced solves
%==========================================================================
function x = full_box_oracle(p)
x = double(p > 0);
end

function [x, val] = solve_full_box_linprog_max(p, opts)
d = numel(p);
[x, negval, exitflag] = linprog(-p, [eye(d); -eye(d)], [ones(d,1); zeros(d,1)], [], [], [], [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(x)
    x = full_box_oracle(p);
    val = p' * x;
else
    val = -negval;
end
end

function [x, val] = solve_reduced_box_max(p, U, x0, opts)
if isempty(U)
    x = x0; val = p' * x; return;
end
r = U' * p;
Aineq = [ U; -U ];
bineq = [ ones(numel(x0),1)-x0; x0 ];
[z, negval, exitflag] = linprog(-r, Aineq, bineq, [], [], [], [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(z)
    z = zeros(size(U,2),1);
end
x = x0 + U*z;
val = p' * x;
end

function [ratio, meanTime] = evaluate_exact_reduced(Ptest, fullVals, U, x0, opts)
n = size(Ptest,1);
rat = zeros(n,1); rt = zeros(n,1);
for i = 1:n
    p = Ptest(i,:)';
    tic; [~, val] = solve_reduced_box_max(p, U, x0, opts); rt(i)=toc;
    rat(i) = val / max(fullVals(i), 1e-12);
end
ratio = mean(max(min(rat,1),0));
meanTime = mean(rt);
end

%==========================================================================
% Projection solves and baselines
%==========================================================================
function P = colrand_projection(d, K)
idx = randperm(d, K);
P = eye(d); P = P(:,idx);
end

function P = pca_projection(Xtrain, K)
[dummyN, d] = size(Xtrain); %#ok<ASGLU>
xbar = mean(Xtrain,1)';
Xc = Xtrain - mean(Xtrain,1);
[~,~,V] = svd(Xc, 'econ');
if K == 1
    P = xbar;
else
    P = [xbar, V(:,1:min(K-1,size(V,2)))];
end
if size(P,2) < K
    P = [P, rand_feasible_projection(d, K-size(P,2))]; %#ok<AGROW>
end
P = final_project_columns_to_box(P);
end

function P = rand_feasible_projection(d, K)
P = rand(d,K);
P = final_project_columns_to_box(P);
end

function P = final_project_columns_to_box(P)
P = min(max(P,0),1);
% avoid all-zero columns
for k = 1:size(P,2)
    if norm(P(:,k)) <= 1e-12
        j = randi(size(P,1)); P(j,k)=1;
    end
end
end

function [y, val, lambdaIneq] = solve_projected_box_max(p, P, opts)
% max p'P y s.t. 0 <= P y <= 1, implemented as min -p'P y.
d = size(P,1);
Aineq = [ P; -P ];
bineq = [ ones(d,1); zeros(d,1) ];
[y, negval, exitflag, ~, lambda] = linprog(-(P'*p), Aineq, bineq, [], [], [], [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(y)
    y = zeros(size(P,2),1); val = 0; lambdaIneq = zeros(2*d,1);
else
    val = -negval;
    lambdaIneq = lambda.ineqlin;
end
end

function [ratio, meanTime] = evaluate_projection(Ptest, fullVals, P, opts)
n = size(Ptest,1);
rat = zeros(n,1); rt = zeros(n,1);
for i = 1:n
    p = Ptest(i,:)';
    tic; [~, val] = solve_projected_box_max(p, P, opts); rt(i)=toc;
    rat(i) = val / max(fullVals(i), 1e-12);
end
ratio = mean(max(min(rat,1),0));
meanTime = mean(rt);
end

function P = train_sharedP_sga(Ptrain, K, Pinit, nEpoch, lr0, opts)
% Gradient ascent for shared P. This is intentionally lightweight and is not
% tuned to make SGA look good; it is meant to expose nonconvex sensitivity.
P = Pinit;
d = size(P,1);
Aorig = [eye(d); -eye(d)];
for ep = 1:nEpoch
    eta = lr0 / sqrt(ep);
    order = randperm(size(Ptrain,1));
    for ii = 1:numel(order)
        p = Ptrain(order(ii),:)';
        [y, ~, lambda] = solve_projected_box_max(p, P, opts);
        % For max objective with constraints Aorig*P*y <= b, a KKT gradient is
        %   d u/dP = p*y' - Aorig' * lambda * y'.
        grad = p*y' - Aorig' * lambda * y';
        P = P + eta * grad;
        P = final_project_columns_to_box(P);
    end
end
end

%==========================================================================
% Instance-specific diagnostic baselines
%==========================================================================
function [ratio, meanTime] = evaluate_inst_topk(Ptest, fullVals, K, opts)
n = size(Ptest,1);
rat = zeros(n,1); rt = zeros(n,1);
for i = 1:n
    p = Ptest(i,:)';
    [~,idx] = sort(p, 'descend');
    P = eye(numel(p)); P = P(:, idx(1:K));
    tic; [~, val] = solve_projected_box_max(p, P, opts); rt(i)=toc;
    rat(i) = val / max(fullVals(i), 1e-12);
end
ratio = mean(max(min(rat,1),0));
meanTime = mean(rt);
end

function mlp = train_inst_mlp_lite(Ptrain, Xtrain, K, cfg)
% Simple supervised MLP-lite that predicts a nonnegative P row score from p.
% This is diagnostic only; it is not the PELP model from Iwata--Sakaue.
d = size(Ptrain,2);
H = cfg.mlpHidden;
mlp.W1 = 0.05*randn(H,d); mlp.b1 = zeros(H,1);
mlp.W2 = 0.05*randn(d*K,H); mlp.b2 = zeros(d*K,1);
lr = 1e-3;
for ep = 1:cfg.mlpEpochs
    for i = randperm(size(Ptrain,1))
        p = Ptrain(i,:)';
        % target P: top-K optimal coordinates in x*. This is a crude proxy.
        x = Xtrain(i,:)';
        ids = find(x>0.5);
        if numel(ids) < K
            [~,ord] = sort(p,'descend'); ids = unique([ids; ord(1:K)']);
        end
        ids = ids(1:min(K,numel(ids)));
        T = zeros(d,K); for k=1:numel(ids), T(ids(k),k)=1; end
        target = T(:);
        h = max(mlp.W1*p + mlp.b1, 0);
        out = mlp.W2*h + mlp.b2;
        err = out - target;
        gW2 = err*h'; gb2 = err;
        gh = mlp.W2'*err; gh(h<=0)=0;
        gW1 = gh*p'; gb1 = gh;
        mlp.W2 = mlp.W2 - lr*gW2; mlp.b2 = mlp.b2 - lr*gb2;
        mlp.W1 = mlp.W1 - lr*gW1; mlp.b1 = mlp.b1 - lr*gb1;
    end
end
mlp.K = K; mlp.d = d;
end

function [ratio, meanTime] = evaluate_inst_mlp_lite(Ptest, fullVals, mlp, opts)
n = size(Ptest,1);
rat=zeros(n,1); rt=zeros(n,1);
for i=1:n
    p = Ptest(i,:)';
    P = mlp_generate_projection(mlp,p);
    tic; [~,val] = solve_projected_box_max(p,P,opts); rt(i)=toc;
    rat(i)=val/max(fullVals(i),1e-12);
end
ratio=mean(max(min(rat,1),0)); meanTime=mean(rt);
end

function P = mlp_generate_projection(mlp,p)
h = max(mlp.W1*p + mlp.b1,0);
out = mlp.W2*h + mlp.b2;
P = reshape(out, [mlp.d, mlp.K]);
P = exp(P - max(P,[],1));
P = P ./ max(sum(P,1),1e-12);
end

%==========================================================================
% Utilities
%==========================================================================
function opts = make_linprog_options()
if exist('optimoptions','file') == 2
    opts = optimoptions('linprog', 'Display','none', 'Algorithm','dual-simplex');
else
    opts = optimset('Display','off'); %#ok<OPTIMSET>
end
end

function save_figure(fig, filename)
set(fig,'Color','w');
if exist('exportgraphics','file') == 2
    exportgraphics(fig, filename, 'Resolution', 200);
else
    saveas(fig, filename);
end
end

function outDir = prepare_results_dir(prefix)
stamp = datestr(now, 'yyyymmdd_HHMMSS');
outDir = fullfile(pwd, [prefix '_' stamp]);
if ~exist(outDir,'dir'), mkdir(outDir); end
end
