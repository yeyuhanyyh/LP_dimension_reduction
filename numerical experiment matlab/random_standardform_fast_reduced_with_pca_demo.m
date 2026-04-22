function random_standardform_fast_reduced_with_pca_demo()
%========================================================================= %
% random_standardform_fast_reduced_with_pca_demo
% -------------------------------------------------------------------------
% Fully random standard-form LP benchmark with BOTH:
%   (i)  a solver-friendly exact reduced LP from learned decision directions;
%   (ii) a PCA baseline built from training optimal solutions.
%
% LP family:
%     min c' x   s.t. A x = b, x >= 0,
% where A is randomly generated and NOT of the planted [I,T] form.
%
% Local reduced LP:
% Around a random reference basis B0, if S is the learned set of relevant
% nonbasic pivot directions, the exact fast deployment problem is
%     min_z r_S(c)' z   s.t.  T_S z <= x_B0, z >= 0,
% where T_S = A_B^{-1} A_S and r_S(c)=c_S-A_S' A_B^{-T} c_B.
%
% PCA baseline:
% We solve the full LP on all training costs, run PCA on the training
% optimizers, and deploy the affine PCA subspace xbar + U_pca z.
%
% Default sampling is i.i.d. local-rare inside the same ball prior C0. This
% makes the comparison fair: train and test are from the same distribution,
% but finite training samples may miss rare relevant pivot directions. The
% FI-based learner uses the prior C0 and can discover such directions.
%========================================================================= %

clc; close all;

cfg.seed       = 20260420;
cfg.m          = 120;        % equality constraints
cfg.d          = 2200;       % variables; choose d >> Ktarget
cfg.Ktarget    = 20;         % nominal intrinsic/local dimension
cfg.nTrain     = 60;
cfg.nTest      = 80;
cfg.nTrial     = 4;
cfg.maxAlg1Its = 2*cfg.Ktarget + 20;
cfg.fiTol      = 1e-9;
cfg.indepTol   = 1e-8;

% Sampling inside C0. iidLocalRare is PCA-unfriendly but train/test i.i.d.
cfg.sampleMode  = 'iidLocalRare'; % 'iidLocalRare' or 'uniformBall'
cfg.pRare       = 0.25;           % per-sample probability of activating a local pivot
cfg.centerNoise = 0.02;           % center noise radius as fraction of rho
cfg.sampleFrac  = 0.98;           % stay inside sampleFrac*rho for rare activations

% Random center reduced-cost spread. Larger -> clearer margin gap.
cfg.reducedCostSpread = 3.0;
cfg.Anoise = 0.20;

if exist('linprog','file') ~= 2
    error('This demo requires MATLAB linprog from Optimization Toolbox.');
end

rng(cfg.seed, 'twister');

rankMat = zeros(cfg.nTrial, cfg.nTrain);
learnDim = zeros(cfg.nTrial,1);
localK = zeros(cfg.nTrial,1);
rhoVec = zeros(cfg.nTrial,1);
originMargin = zeros(cfg.nTrial,1);
pcaDim = zeros(cfg.nTrial,1);
rareTrainCoverage = zeros(cfg.nTrial,1);
rareTestCoverage = zeros(cfg.nTrial,1);

rtFull = zeros(cfg.nTrial,cfg.nTest);
rtFast = zeros(cfg.nTrial,cfg.nTest);
rtGeneric = zeros(cfg.nTrial,cfg.nTest);
rtPCA = zeros(cfg.nTrial,cfg.nTest);

gapFast = zeros(cfg.nTrial,cfg.nTest);
gapGeneric = zeros(cfg.nTrial,cfg.nTest);
gapPCA = zeros(cfg.nTrial,cfg.nTest);

capFast = zeros(cfg.nTrial,1);
capPCA = zeros(cfg.nTrial,1);

for tr = 1:cfg.nTrial
    rng(cfg.seed + tr, 'twister');
    fam = generate_random_standardform_family(cfg);

    [Ctrain, trainActive] = sample_costs_for_family(cfg.nTrain, fam, cfg);
    [Ctest,  testActive]  = sample_costs_for_family(cfg.nTest,  fam, cfg);

    % PCA baseline: solve training optima and compute affine PCA subspace.
    XtrainOpt = zeros(cfg.d, cfg.nTrain);
    for ii = 1:cfg.nTrain
        [xii, ~] = solve_full_standard_lp(Ctrain(ii,:)', fam);
        XtrainOpt(:,ii) = xii;
    end
    [xPCA0, Upca, pcaDim(tr)] = pca_subspace_from_training_optima(XtrainOpt, cfg.Ktarget);

    % Our Stage-I learner.
    [Dfinal, Slearn, info] = alg2_cumulative_ball(Ctrain, fam, cfg);
    UlearnOrth = orth(Dfinal);  % diagnostic generic-U; fast pivot uses Slearn.

    rankMat(tr,:) = info.rankAfterSample(:)';
    learnDim(tr) = size(Dfinal,2);
    localK(tr) = numel(fam.Slocal);
    rhoVec(tr) = fam.rho;
    originMargin(tr) = norm(fam.c0) - fam.rho;
    rareTrainCoverage(tr) = numel(unique(trainActive(trainActive>0)));
    rareTestCoverage(tr)  = numel(unique(testActive(testActive>0)));

    fprintf('\nTrial %d/%d\n', tr, cfg.nTrial);
    fprintf('  d=%d, m=%d, target K=%d, local K=%d, learned dim=%d\n', cfg.d, cfg.m, cfg.Ktarget, numel(fam.Slocal), size(Dfinal,2));
    fprintf('  rho=%.3e, ||c0||-rho=%.3e, |Slearn|=%d\n', fam.rho, norm(fam.c0)-fam.rho, numel(Slearn));
    fprintf('  PCA nominal K=%d, PCA effective dim=%d, rare coverage train=%d/%d, test=%d/%d\n', ...
        cfg.Ktarget, pcaDim(tr), rareTrainCoverage(tr), numel(fam.Slocal), rareTestCoverage(tr), numel(fam.Slocal));

    % Warm-up to reduce first-call overhead.
    for w = 1:min(2,cfg.nTest)
        c = Ctest(w,:)';
        solve_full_standard_lp(c, fam);
        solve_reduced_pivot_lp(c, fam, Slearn);
        solve_reduced_affine_lp(c, fam, fam.x0, UlearnOrth);
        solve_reduced_affine_lp(c, fam, xPCA0, Upca);
    end

    for t = 1:cfg.nTest
        c = Ctest(t,:)';

        tic; [~, vFull] = solve_full_standard_lp(c, fam); rtFull(tr,t) = toc;
        tic; [~, vFast] = solve_reduced_pivot_lp(c, fam, Slearn); rtFast(tr,t) = toc;
        tic; [~, vGen]  = solve_reduced_affine_lp(c, fam, fam.x0, UlearnOrth); rtGeneric(tr,t) = toc;
        tic; [~, vPCA]  = solve_reduced_affine_lp(c, fam, xPCA0, Upca); rtPCA(tr,t) = toc;

        gapFast(tr,t) = abs(vFast - vFull) / max(1,abs(vFull));
        gapGeneric(tr,t) = abs(vGen - vFull) / max(1,abs(vFull));
        gapPCA(tr,t) = abs(vPCA - vFull) / max(1,abs(vFull));
    end

    capFast(tr) = compute_improvement_capture(Ctest, fam, @(cc) solve_reduced_pivot_lp(cc, fam, Slearn));
    capPCA(tr)  = compute_improvement_capture(Ctest, fam, @(cc) solve_reduced_affine_lp(cc, fam, xPCA0, Upca));

    fprintf('  median times: full=%.4g, fast=%.4g, genericU=%.4g, PCA=%.4g\n', ...
        median(rtFull(tr,:)), median(rtFast(tr,:)), median(rtGeneric(tr,:)), median(rtPCA(tr,:)));
    fprintf('  speedup fast/full=%.2fx; max gaps fast=%.2e, PCA=%.2e\n', ...
        median(rtFull(tr,:)./rtFast(tr,:)), max(gapFast(tr,:)), max(gapPCA(tr,:)));
    fprintf('  improvement capture: ours=%.3f, PCA=%.3f\n', capFast(tr), capPCA(tr));
end

% ----------------------------- plots ------------------------------------
[mRank, ciRank] = mean_ci90(rankMat);
fig1 = figure('Color','w'); hold on; box on; grid on;
errorbar(1:cfg.nTrain, mRank, ciRank, 'LineWidth',1.5);
yline(mean(localK), '--k', 'LineWidth',1.2);
xlabel('Training sample index i');
ylabel('t_i = dim(\hat W_i)');
title('Random standard-form LP: Stage I learned dimension');
legend({'mean \pm 90% CI', 'mean local K'}, 'Location','southeast');

fig2 = figure('Color','w'); box on; grid on;
bar([median(rtFull(:)), median(rtFast(:)), median(rtGeneric(:)), median(rtPCA(:))]);
set(gca, 'XTickLabel', {'FullStdLP', 'OursFastPivot', 'OursGenericU', 'PCA'});
ylabel('Median solve time (seconds)');
title(sprintf('Runtime comparison: d=%d, m=%d, K≈%d', cfg.d, cfg.m, round(mean(localK))));

fig3 = figure('Color','w'); box on; grid on;
bar([max(gapFast(:)), max(gapGeneric(:)), max(gapPCA(:))]);
set(gca, 'XTickLabel', {'OursFastPivot', 'OursGenericU', 'PCA'});
ylabel('Max relative objective gap vs full LP');
title('Objective exactness / quality check');

fig4 = figure('Color','w'); hold on; box on; grid on;
scatter(localK, median(rtFull,2)./median(rtFast,2), 70, 'filled');
xlabel('Local relevant dimension K');
ylabel('Median speedup FullStdLP / OursFastPivot');
title('Speedup by trial');

fig5 = figure('Color','w'); box on; grid on;
bar([mean(capFast), mean(capPCA)]);
set(gca, 'XTickLabel', {'OursFastPivot', 'PCA'});
ylabel('Improvement capture, higher is better');
title(sprintf('Decision-quality comparison at nominal K=%d', cfg.Ktarget));

fig6 = figure('Color','w'); box on; grid on;
bar([mean(learnDim), mean(pcaDim), mean(rareTrainCoverage), mean(rareTestCoverage)]);
set(gca, 'XTickLabel', {'learned dim', 'PCA eff. dim', 'train rare coverage', 'test rare coverage'});
ylabel('Dimension / coverage');
title('PCA visibility diagnostics');

outDir = prepare_results_dir('random_standardform_fast_reduced_with_pca_results');
save_figure(fig1, fullfile(outDir, 'stage1_mean_dimension.png'));
save_figure(fig2, fullfile(outDir, 'runtime_bars.png'));
save_figure(fig3, fullfile(outDir, 'objective_gap_bars.png'));
save_figure(fig4, fullfile(outDir, 'speedup_scatter.png'));
save_figure(fig5, fullfile(outDir, 'improvement_capture_bars.png'));
save_figure(fig6, fullfile(outDir, 'pca_visibility_diagnostics.png'));
save(fullfile(outDir, 'summary.mat'), 'cfg', 'rankMat', 'learnDim', 'localK', 'rhoVec', ...
    'originMargin', 'rtFull', 'rtFast', 'rtGeneric', 'rtPCA', ...
    'gapFast', 'gapGeneric', 'gapPCA', 'capFast', 'capPCA', 'pcaDim', ...
    'rareTrainCoverage', 'rareTestCoverage');

fprintf('\nSaved results to: %s\n', outDir);
end

%========================================================================= %
% Random family generation
%========================================================================= %
function fam = generate_random_standardform_family(cfg)
m = cfg.m; d = cfg.d; nN = d-m;
if nN <= cfg.Ktarget + 5
    error('Need d-m substantially larger than Ktarget.');
end

B = sort(randperm(d, m));
N = setdiff(1:d, B);

AB = abs(rand(m,m) + cfg.Anoise*randn(m,m)) + 0.05*eye(m);
while rcond(AB) < 1e-6
    AB = abs(rand(m,m) + cfg.Anoise*randn(m,m)) + 0.05*eye(m);
end
AN = abs(rand(m,nN) + cfg.Anoise*randn(m,nN)) + 0.02*rand(m,nN);
A = zeros(m,d);
A(:,B) = AB;
A(:,N) = AN;

xB0 = 0.5 + rand(m,1);
b = AB*xB0;
x0 = zeros(d,1); x0(B) = xB0;

% Center cost in the cone of B. Local reduced costs are random positive.
cB = 5 + randn(m,1);
lambda = AB' \ cB;
rRaw = exp(cfg.reducedCostSpread*randn(nN,1));
rRaw = rRaw / median(rRaw);
cN = AN' * lambda + rRaw;

c0 = zeros(d,1); c0(B) = cB; c0(N) = cN;
T = AB \ AN;
DeltaN = zeros(d,nN);
tau = zeros(nN,1);
for k = 1:nN
    delta = zeros(d,1);
    delta(B) = -T(:,k);
    delta(N(k)) = 1;
    DeltaN(:,k) = delta;
    tau(k) = max(0, rRaw(k)) / max(norm(delta),1e-12);
end

[tauSort, order] = sort(tau, 'ascend');
K = min(cfg.Ktarget, nN-1);
if tauSort(K+1) > tauSort(K)
    rho = 0.5*(tauSort(K) + tauSort(K+1));
else
    rho = tauSort(K)*1.05;
end
if rho >= 0.45*norm(c0)
    rho = 0.45*norm(c0);
end
Slocal = order(tau <= rho + 1e-12);

fam.A = A; fam.b = b; fam.B0 = B(:); fam.N0 = N(:); fam.AB = AB; fam.AN = AN;
fam.xB0 = xB0; fam.x0 = x0; fam.c0 = c0; fam.rho = rho;
fam.T = T; fam.DeltaN = DeltaN; fam.tau = tau; fam.Slocal = Slocal;
fam.cB0 = cB; fam.rN0 = rRaw;
end

%========================================================================= %
% Sampling
%========================================================================= %
function [C, activeIdx] = sample_costs_for_family(n, fam, cfg)
if strcmp(cfg.sampleMode, 'uniformBall')
    C = sample_costs_from_ball(n, fam.c0, fam.rho*cfg.sampleFrac, numel(fam.c0));
    activeIdx = zeros(n,1);
    return;
end

% iidLocalRare: most samples stay near c0; rare samples cross one local
% relevant facet. Train and test call the same function, so there is no
% train/test shift.
d = numel(fam.c0);
C = zeros(n,d);
activeIdx = zeros(n,1);
for i = 1:n
    c = fam.c0 + cfg.centerNoise*fam.rho*rand_unit_vector(d);
    if rand < cfg.pRare && ~isempty(fam.Slocal)
        s = fam.Slocal(randi(numel(fam.Slocal)));
        q = fam.DeltaN(:,s);
        qn = q / max(norm(q),1e-12);
        tau = fam.tau(s);
        lo = min(cfg.sampleFrac*fam.rho, tau + 1e-4*fam.rho);
        hi = cfg.sampleFrac*fam.rho;
        if hi > lo
            amount = lo + (hi-lo)*rand;
        else
            amount = hi;
        end
        c = fam.c0 - amount*qn;
        activeIdx(i) = s;
    end
    % Numerical safety: project back to the ball if tiny noise overshoots.
    diff = c - fam.c0;
    nr = norm(diff);
    if nr > cfg.sampleFrac*fam.rho
        c = fam.c0 + (cfg.sampleFrac*fam.rho/nr)*diff;
    end
    C(i,:) = c';
end
end

function C = sample_costs_from_ball(n, c0, radius, d)
C = zeros(n,d);
for i = 1:n
    C(i,:) = (c0 + radius*rand_unit_vector(d)*rand^(1/d))';
end
end

function u = rand_unit_vector(d)
u = randn(d,1);
u = u/max(norm(u),1e-12);
end

%========================================================================= %
% PCA baseline
%========================================================================= %
function [xbar, U, effRank] = pca_subspace_from_training_optima(XtrainOpt, K)
xbar = mean(XtrainOpt,2);
Xc = XtrainOpt - xbar;
[Uall,S,~] = svd(Xc, 'econ');
sv = diag(S);
if isempty(sv) || sv(1) <= 1e-12
    effRank = 0; U = zeros(size(XtrainOpt,1),0); return;
end
effRankAll = sum(sv > 1e-8*sv(1));
effRank = min([K, effRankAll, size(Uall,2)]);
U = Uall(:,1:effRank);
end

%========================================================================= %
% Algorithm 2 cumulative learner with ball-prior closed-form FI
%========================================================================= %
function [Dfinal, Slearn, info] = alg2_cumulative_ball(Ctrain, fam, cfg)
[nTrain,d] = size(Ctrain);
D = zeros(d,0); Slearn = zeros(0,1); rankAfterSample = zeros(nTrain,1);
for i = 1:nTrain
    c = Ctrain(i,:)';
    [D, Slearn] = alg1_pointwise_ball(c, D, Slearn, fam, cfg);
    rankAfterSample(i) = size(D,2);
end
Dfinal = D; info.rankAfterSample = rankAfterSample;
end

function [D, Slearn] = alg1_pointwise_ball(cAnchor, Dinit, SlearnInit, fam, cfg)
D = Dinit; Slearn = SlearnInit;
for it = 1:cfg.maxAlg1Its
    [x, ~] = solve_full_standard_lp(cAnchor, fam);
    B = recover_basis_from_solution(x, fam, cfg);
    [Delta, Ncur] = compute_deltas_for_basis(fam.A, B);
    violated = false(numel(Ncur),1); alphaVal = inf(numel(Ncur),1);
    for k = 1:numel(Ncur)
        q = Delta(:,k);
        [mVal, ~] = min_linear_over_ball_fiber(q, D, cAnchor, fam.c0, fam.rho);
        cinVal = q'*cAnchor;
        if mVal < -cfg.fiTol
            violated(k) = true;
            denom = cinVal - mVal;
            if denom <= 1e-14, alphaVal(k)=0; else, alphaVal(k)=cinVal/denom; end
        end
    end
    if ~any(violated), break; end
    ids = find(violated); [~,loc] = min(alphaVal(ids)); pick = ids(loc);
    qPick = Delta(:,pick);
    [D, wasAdded] = append_direction_if_new(D, qPick, cfg.indepTol);
    if wasAdded
        idxInN0 = find(fam.N0 == Ncur(pick), 1);
        if ~isempty(idxInN0) && ~ismember(idxInN0, Slearn)
            Slearn(end+1,1) = idxInN0; %#ok<AGROW>
        end
    else
        break;
    end
end
Slearn = sort(Slearn(:));
end

function [mVal, cOut] = min_linear_over_ball_fiber(q, D, cAnchor, c0, rho)
d = numel(q);
if isempty(D), Q=zeros(d,0); else, Q=orth(D); end
proj = Q*(Q'*(cAnchor-c0));
centerFib = c0 + proj;
radEff = sqrt(max(rho^2 - norm(proj)^2, 0));
qPerp = q - Q*(Q'*q);
if norm(qPerp) <= 1e-12
    cOut = centerFib; mVal = q'*cOut;
else
    cOut = centerFib - radEff*qPerp/norm(qPerp); mVal = q'*cOut;
end
end

%========================================================================= %
% Basis recovery and deltas
%========================================================================= %
function B = recover_basis_from_solution(x, fam, cfg)
[~,idx] = sort(x, 'descend'); B = sort(idx(1:size(fam.A,1)));
if rank(fam.A(:,B), cfg.indepTol) < size(fam.A,1)
    B = fam.B0(:)';
end
end

function [Delta, N] = compute_deltas_for_basis(A, B)
[m,d] = size(A); B = B(:)'; N = setdiff(1:d, B); AB = A(:,B); T = AB\A(:,N);
Delta = zeros(d,numel(N));
for k = 1:numel(N)
    delta = zeros(d,1); delta(B) = -T(:,k); delta(N(k)) = 1; Delta(:,k) = delta;
end
end

function [Dnew, wasAdded] = append_direction_if_new(D, q, tol)
q = q/max(norm(q),1e-12);
if isempty(D), Dnew=q; wasAdded=true; return; end
coeff = (D'*D)\(D'*q); res = q-D*coeff;
if norm(res) <= tol*max(1,norm(q)), Dnew=D; wasAdded=false;
else, Dnew=[D, res/norm(res)]; wasAdded=true; end %#ok<AGROW>
end

%========================================================================= %
% LP solvers
%========================================================================= %
function [x, val] = solve_full_standard_lp(c, fam)
opts = make_linprog_options();
[x,val,exitflag] = linprog(c, [], [], fam.A, fam.b, zeros(numel(c),1), [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(x), error('Full standard-form LP failed.'); end
end

function [x, val] = solve_reduced_pivot_lp(c, fam, S)
if isempty(S), x=fam.x0; val=c'*x; return; end
opts = make_linprog_options(); B=fam.B0; N=fam.N0; TS=fam.T(:,S); origS=N(S);
y = fam.AB' \ c(B); rS = c(origS) - fam.A(:,origS)'*y;
[z,~,exitflag] = linprog(rS, TS, fam.xB0, [], [], zeros(numel(S),1), [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(z), z=zeros(numel(S),1); end
x = fam.x0; x(B)=fam.xB0-TS*z; x(origS)=z; val=c'*x;
end

function [x, val] = solve_reduced_affine_lp(c, fam, xAnchor, U)
if isempty(U), x=xAnchor; val=c'*x; return; end
opts = make_linprog_options(); r=U'*c; Aineq=-U; bineq=xAnchor;
Aeq = fam.A*U; beq = fam.b - fam.A*xAnchor;
[z,~,exitflag] = linprog(r, Aineq, bineq, Aeq, beq, -inf(size(U,2),1), [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(z), z=zeros(size(U,2),1); end
x = xAnchor + U*z; val = c'*x;
end

%========================================================================= %
% Metrics and plots
%========================================================================= %
function cap = compute_improvement_capture(Ctest, fam, solverHandle)
impFull = 0; impMethod = 0;
for i = 1:size(Ctest,1)
    c = Ctest(i,:)';
    [~,vFull] = solve_full_standard_lp(c, fam);
    [~,vMeth] = solverHandle(c);
    vBase = c'*fam.x0;
    impFull = impFull + max(0, vBase - vFull);
    impMethod = impMethod + max(0, vBase - vMeth);
end
cap = impMethod / max(impFull,1e-12);
cap = max(0, min(1.05, cap));
end

function [m,ci] = mean_ci90(X)
m = mean(X,1);
if size(X,1)==1, ci=zeros(size(m)); else, ci=1.645*std(X,0,1)/sqrt(size(X,1)); end
end

function opts = make_linprog_options()
if exist('optimoptions','file') == 2
    try, opts = optimoptions('linprog','Display','none','Algorithm','dual-simplex');
    catch, opts = optimoptions('linprog','Display','none'); end
else, opts = optimset('Display','off'); end %#ok<OPTIMSET>
end

function save_figure(fig, filename)
set(fig,'Color','w');
if exist('exportgraphics','file') == 2, exportgraphics(fig, filename, 'Resolution', 200);
else, saveas(fig, filename); end
end

function outDir = prepare_results_dir(prefix)
stamp = datestr(now, 'yyyymmdd_HHMMSS'); outDir = fullfile(pwd, [prefix '_' stamp]);
if ~exist(outDir,'dir'), mkdir(outDir); end
end
