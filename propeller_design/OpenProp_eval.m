
% =========================================================================
% OpenProp_v3.3.4
% Last modified: 1/13/2021 @hotel Victor 
% =========================================================================

% ========================= Initiate OpenProp ======================

% ================================ execute ================================
%function OpenProp_edit
function [sim_out] = OpenProp_eval(designdata)
    %newPlots;
    
   %thrust,vel_ship,rpm,dia,eff,cd1,cd2,cd3,cd4,cd5,cd6,cd7,cd8,cd9,cd10
   
    %designdata
    global Plots PlotPanels Toggle OnDesignValues ConversionValues systemToggle;

    global OpenPropDirectory SpecificationsValues DuctValues FlagValues FoilValues Filename...
           XR_in XCoD_in XCD_in VAI_in ri_in VTI_in Xt0oD_in skew0_in rake0_in...
           Meanline_cell Thickness_cell; % CavValues

    global pt

    %{
    filename = get(Filename,'string');                       % Filename prefix
 

    while ~isempty(rest)
        [CurrentDirectory,rest] = strtok(rest,'/');

        if strcmp(CurrentDirectory,OpenPropDirectory)

            if isempty(rest)

                % you are in /OpenPropDirectory/
                mkdir(['./',filename])
                   cd(['./',filename])
                   addpath ../SourceCode

            elseif strcmp(rest(2:end),filename)
                % already in /OpenPropDirectory/filename/
                addpath ../SourceCode
                rest = [];

            elseif strcmp(rest(2:end),'SourceCode')

                mkdir(['../',filename])
                   cd(['../',filename])
                   addpath ../SourceCode
                rest = [];

            else
                % you are in /OpenPropDirectory/wrongfolder
                disp('ERROR2: Must start OpenProp from the root directory.')
                return
            end
        end
    end
    % -------------------------------------------------------------------------
    %}

    % --------------------------- Design parameters(fixed) ---------------------------

    Z           = 3  ;      % number of blades
   
    ITER        = 40;       % number of iterations in analysis
    Rhv         = 0.1;   	% hub vortex radius / hub radius

    TAU         = 1; %str2double(get(DuctValues(1),'string'));      % Thrust ratio
    CDd         = 0.008; %str2double(get(DuctValues(2),'string'));      % Duct section drag coefficient

    %Dhub        = 0.8534 ;      % hub diameter [m]
    rho         = 1000 	;    % Sea water density [kg/m^3]
    Mp          = 20 ;	    % number of vortex panels over the radius
    Np          = 20 ;	    % Number of points over the chord [ ]

    
    
    % ----Flags fixed---------------------------------

    Propeller_flag	= 1; % get(FlagValues(1),'value');               % 0 == turbine, 1 == propeller
    Hub_flag  	    = 1 ;% get(FlagValues(3),'value');                   % 0 == no hub, 1 == hub
    Duct_flag	    = 0; % get(FlagValues(4),'value');                   % 0 == no duct, 1 == duct

    Chord_flag	    = 0 ;% get(FlagValues(5),'value');                   % ** CHORD OPTIMIZATION FLAG **

    Viscous_flag	= 1 ;% get(FlagValues(6),'value');               % 0 == viscous forces off (CD = 0), 1 == viscous forces on
    Plot_flag       = 0 ;% get(FlagValues(7),'value');               % 0 == do not display plots, 1 == display plots



    Make2Dplot_flag = 0 ;%get(FlagValues(8),'value');               % 0 == do not make a 2D plot of the results, 1 == make plot
    Make3Dplot_flag = 0 ;% get(FlagValues(8),'value');               % 0 == do not make a 3D plot of the results, 1 == make plot

    Analyze_flag	= 0 ;% get(FlagValues(9),'value');

    % Cav_flag	= get(FlagValues(10),'value');                   % 0 == do not run cavitation mapping, 1 == run cavitation mapping
    
    Meanline_index  = 1; % 1 for yes, 0 for false
    Meanline        = 'NACA a=0.8'; %      % Meanline form

    Thickness_index	= 1; %1 for yes, 0 for false
    Thickness       = 'NACA 65A010'; % char(Thickness_cell(Thickness_index));            % Thickness form

    XCD     	=  [0.0090; 0.0090; 0.0090; 0.0090; 0.0090; 0.0090; 0.0090; 0.0090; 0.0090; 0.0090] ;str2double(get(XCD_in, 'string'));            	% section drag coefficient
    XR          = [0.2000; 0.3000; 0.4000; 0.5000;  0.6000; 0.7000; 0.8000; 0.9000; 0.9500; 1.0000];   %str2double(get(XR_in,  'string'));             	% radius / propeller radius
   
    ri          =  [NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN] ; %str2double(get(ri_in, 'string'));
    VAI         = [NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN]  ;           % axial      inflow velocity / ship velocity
    VTI         = [NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN] ;       	% tangential inflow velocity / ship velocity

    ri  = ri (~isnan(ri));
    VAI = VAI(~isnan(VAI));
    VTI = VTI(~isnan(VTI));

    skew0       = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0] ;        	% skew
    rake0       = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0]  ;    	% rake
    Xt0oD       = [0.0329; 0.0281; 0.0239; 0.0198; 0.0160; 0.0125; 0.0091; 0.0060; 0.0045; 0] ;% max section thickness / chord

    % --------------------------- Design parameters(set it) -------------------------------
    % -------------------------------------------------------------------------------------- 
                            
    N           =   designdata(3);                % set propeller speed [RPM] Coming from designer
    D           =  designdata(4)  ;	              % set propeller diameter [m]  
    Dhub        =  0.17*D  ;
    THRUST           = designdata(1) ;            % set Thrust required [Newton]
    Vs= designdata(2) ;                           % set velocity 

    %  Blade 2D section properties ----------------------
    
    XCoD        = designdata(5:14) ;  ; % chord / diameter
    XCLmax      = XCoD ; %If cord optimisation is off maximum lift coefficient (for chord optimization)
  
    %--------------------------------------------------------------------------------------------------
    % input ::[THRUST ,velocity(Vs),rpm,diameter] ==> [CoD ] (later beta)
    %--------------------------------------------------------------------------------------------------
    % for beta => fix CoD and then change 
    %--------------------------------------------------------------------------------------------------  
 

    % ----------------------- Compute derived quantities ----------------------

    n           = N/60;                                         % ** propeller speed [rev/s] = Vs/(Js*D) = N/60
    Js          = Vs/(n*D);                                     % ** Js = Vs/(n*D) ,  advance coefficient
    KT          = THRUST/(rho*n^2*D^4);                        	% ** KT = THRUST/(rho*n^2*D^4)
    L           = pi/Js;                                        % tip speed ratio

    R           = D/2;                                          % propeller radius [m]
    Rhub        = Dhub/2;                                       % hub radius [m]
    Rhub_oR     = Rhub/R;

    CTDES       = THRUST/(0.5*rho*Vs^2*pi*R^2);                 % CT thrust coefficient required          

    % dVs         = dV/Vs;                                        % axial inflow variation / Vs

    
    
    % *************************************************************************
    % *************************************************************************
    input.part1      = '------ Performance inputs ------';
    input.Z          = Z;           % [1 x 1], [ ] number of blades
    input.N          = N;           % propeller speed [RPM]
    input.D          = D;           % propeller diameter [m]  
    input.Vs         = Vs;          % [1 x 1], [m/s] ship speed
    input.Js         = Js;          % [1 x 1], [ ] advance coefficient, Js = Vs/nD = pi/L
    input.L          = L;           % [1 x 1], [ ] tip speed ratio, L = omega*R/V
    input.THRUST     = THRUST;      % required thrust [N]
    input.CTDES      = CTDES;       % [1 x 1], [ ] desired thrust coefficient
    input.TAU        = TAU;         % Thrust ratio
    
    input.part2      = '------ Geometry inputs ------';
    input.Mp         = Mp;          % [1 x 1], [ ] number of blade sections
    input.Np         = Np;          % [1 x 1], [ ] number of points along the chord
    input.R          = R;           % [1 x 1], [m] propeller radius
    input.Rhub       = Rhub;        % [1 x 1], [m] hub radius
    input.XR         = XR;          % [length(XR) x 1], [ ] input radius/propeller radiusat XR
    input.XCD        = XCD;         % [length(XR) x 1], [ ] input drag coefficient       at XR
    input.XCoD       = XCoD;        % [length(XR) x 1], [ ] input chord / diameter       at XR
    input.Xt0oD      = Xt0oD;       % [length(XR) x 1], [ ] input thickness / chord      at XR 
    input.skew0      = skew0;       % [length(XR) x 1], [ ] input skew  [deg]      at XR 
    input.rake0      = rake0;       % [length(XR) x 1], [ ] input rake X/D       at XR 
    input.Meanline   = Meanline;    % 2D section meanline  flag
    input.Thickness  = Thickness;   % 2D section thickness flag 
    input.XCLmax     = XCLmax;

    if ~isempty(ri)  , input.ri  = ri;  end
    if ~isempty(VAI) , input.VAI = VAI; end        % [length(XR) x 1], [ ] input axial inflow velocity  at XR
    if ~isempty(VTI) , input.VTI = VTI; end        % [length(XR) x 1], [ ] input swirl inflow velocity  


    input.Rduct     = R;
    input.Cduct     = D/2;
    input.CDd       = CDd;

    input.part3      = '------ Computational inputs ------';
    input.Propeller_flag  = Propeller_flag; % 0 == turbine, 1 == propeller
    input.Viscous_flag    = Viscous_flag;   % 0 == viscous forces off (CD = 0), 1 == viscous forces on
    input.Hub_flag        = Hub_flag;       % 0 == no hub, 1 == hub
    input.Duct_flag       = Duct_flag;      % 0 == no duct, 1 == duct
    input.Plot_flag       = Plot_flag;      % 0 == do not display plots, 1 == display plots
    input.Chord_flag      = Chord_flag;     % 0 == do not optimize chord lengths, 1 == optimize chord lengths

    input.Make2Dplot_flag = Make2Dplot_flag;
    input.Make3Dplot_flag = Make3Dplot_flag;
    % input.Make_Rhino_flag = Make_Rhino_flag;
    input.ITER            = ITER;           % [ ] number of iterations
    input.Rhv              = Rhv;         % [1 x 1], [ ] hub vortex radius / hub radius

    input.part4      = '------ Cavitation inputs ------';
    input.rho        = rho;         % [1 x 1], [kg/m^3] fluid density
    % input.dVs        = dVs;         % [1 x 1], [ ] ship speed variation / ship speed
    % input.H          = H;           % [1 x 1]

    input.part5      = '------ Duct inputs ------';


    % ---------------------------- Pack up propeller/turbine data structure, pt
  %  pt.filename = filename; % (string) propeller/turbine name
    pt.date     = date;     % (string) date created
  % pt.notes    = ' ';    % (string or cell matrix)   notes
    pt.input    = input;    % (struct) input parameters
    pt.design   = [];       % (struct) design conditions
    pt.geometry = [];       % (struct) design geometry
    pt.states	= [];       % (struct) off-design state analysis

    pt.input.GUI_flag = 1;

   
    
    % ---------------------------------------------------------------------
    % Perform design optimization
    pt.design   = EppsOptimizer(input);
    % ---------------------------------------------------------------------
    
   
    % ---------------------------------------------------------------------
    % Set On Design Performance values

    pt.design.Q = pt.design.CQ * 0.5 * rho * Vs^2 * pi*D^2/4 * D/2; % [Nm]  torque

    omega = 2*pi*n; % [rad/s]

    pt.design.P = pt.design.Q * omega;

    
    KT_after_sim=[pt.design.KT ];
    sim_out=[pt.design.EFFY ];
    if abs(KT-KT_after_sim) <=0.001 
       sim_out=[pt.design.EFFY ];
    else
        sim_out=0.01;
    %sim_out=[cod1,cod2,cod3,cod4,cod5,cod6,cod7,cod8,cod9,cod10 ]
    
    % ---------------------------------------------------------------------
    % Determine propeller geometry
    if Make2Dplot_flag == 1 | Make3Dplot_flag == 1
        pt.geometry = Geometry(pt);
    end
    % ---------------------------------------------------------------------
   
end
% =========================================================================




