# Unsignalized Intersection Control — Game Theory + DRL + CARLA

English / 中文

---

English Version
===============

Summary
-------
This project implements a unified control system for unsignalized intersections using a combination of distributed auctions, game-theoretic deadlock resolution (Nash-based), platoon-aware traffic management, and configurable DRL-tunable parameters. It is built on the CARLA simulator and is intended for research on coordination strategies for mixed traffic (single vehicles + platoons).

Key ideas
---------
- Use a decentralized auction engine to compute a priority ordering among participants (single vehicles and platoons).
- Resolve conflicts and potential deadlocks using a Nash-based deadlock solver that suggests conflict-aware actions (go/wait).
- Enforce auction/Nash decisions via a traffic controller that interacts with CARLA's Traffic Manager to manipulate vehicle behaviors (speed difference, following distance, ignore-lights/vehicles percentages).
- Support platoon-level control: the system can apply leader and follower-specific control parameters to preserve formation and improve throughput.
- Collect per-vehicle acceleration and control history using simulation time for accurate metrics and post-analysis.

Architecture & Major Components
-------------------------------
- CARLA (simulator): world, actors, traffic manager.
- ScenarioManager: scenario setup, spawn points, visual debug overlays, timers.
- StateExtractor: extracts vehicle states (id, location, velocity, is_junction, etc.) for use by other modules.
- PlatoonManager: groups vehicles into platoons, provides leader/follower info and navigation assistance.
- DecentralizedAuctionEngine: participants bid and the engine returns a ranked list of winners with bids and suggested conflict actions.
- DeadlockNashSolver: game-theoretic solver that examines a conflict graph/time-window and proposes actions to avoid deadlock.
- TrafficController: applies control parameters to CARLA actors according to auction/Nash results; tracks control history, acceleration metrics, and exit statistics.

Data Flow
---------
1. ScenarioManager runs and advances the CARLA world.
2. StateExtractor collects vehicle states each tick.
3. PlatoonManager updates platoon membership and navigation states.
4. AuctionEngine receives the current states and platoon info, computes bids and candidate ordering.
5. Nash solver refines conflicting decisions; auction engine returns winners with conflict_action.
6. TrafficController applies controls to vehicles/platoons using Traffic Manager API.
7. TrafficController collects acceleration samples, control timestamps (simulation time), and maintains history for metrics and throughput calculations.

Methodology
-----------
- Auction-based prioritization: participants submit bids that reflect urgency or learned policy values. The decentralized engine ranks participants and couples bids with candidate actions.
- Nash deadlock resolution: when multiple participants conflict, solve a small game to find stable strategies (go/wait combinations) that minimize deadlock risk while respecting bids and a configurable conflict time window.
- Platoon-aware control: leaders and followers receive different TM parameters to keep formation (tighter following, different speed adjustments). If a platoon member is already in-transit, preserve continuity by protecting its 'go' action.
- Acceleration filtering & metrics: record per-vehicle positive/negative/absolute acceleration using simulation timestamps; apply truncation and optional median filtering to reduce noise before averaging.

Configuration & DRL integration
-------------------------------
- DRL-tunable parameters are centralized (example: CONFLICT_TIME_WINDOW). The controller and Nash solver accept configuration updates for experimentation and training loops.
- An optional "max_go_agents" constraint exists for experiments that limit simultaneous go decisions; can be None for unrestricted operation.
- Use SimulationConfig for constants (intersection center, detection radii, PRINT_INTERVAL, FIXED_DELTA_SECONDS, etc.).

Running the Simulation
----------------------
Prerequisites:
- CARLA simulator (matching Python client egg built for your platform)
- Python dependencies used by modules (see project's requirements or add as needed)

Basic steps:
1. Ensure CARLA egg is available under the project (main looks up carla-*linux-x86_64.egg on Linux).
2. Launch CARLA server.
3. Run main.py:
   python main.py
4. The main loop ticks the world, updates modules periodically, and prints unified status (auction order, platoon stats, controller stats).

Important outputs:
- Periodic console report: auction top ranks, controller status, platoon coordination, and detailed statistics on exit count and acceleration averages.
- Final report on termination: total vehicles controlled, vehicles exited, average positive/negative/absolute accelerations, throughput (vehicles/hour), and collision report.

Metrics and Logging
-------------------
- TrafficController tracks:
  - total_vehicles_controlled (cumulative)
  - vehicles_exited_intersection (count, sim-time stamped)
  - control_history (enter & exit sim timestamps and actions)
  - acceleration_data: separate storage for positive, negative, and absolute samples per vehicle
- Filtering and robustness:
  - min_time_delta to skip noisy samples
  - max_acceleration to clamp unrealistic spikes
  - optional median filter window to reduce outliers

Extensibility & Tips
--------------------
- Add DRL training loop that periodically updates DRLConfig and calls traffic_controller.update_max_go_agents(...) or other setters.
- Extend AuctionEngine/DeadlockNashSolver interfaces to return richer diagnostic info for training reward computations.
- Adjust TM parameters in _get_control_params_by_rank_and_action for experiment variants (safety vs throughput trade-offs).

Troubleshooting
---------------
- CARLA egg not found: ensure the right egg is present under carla_l or modify main.py egg discovery logic.
- Timing mismatch: simulations use both wall-clock and CARLA simulation timestamps; metrics prefer sim timestamps for determinism.
- If vehicles appear unresponsive, inspect traffic manager calls and ensure actor IDs match and actors are alive.

License & Contributing
----------------------
- Add appropriate license file (e.g., MIT) and contribution guidelines. Keep modules modular for unit testing and DRL integration.

---

中文版本
=======

概要
----
本项目在 CARLA 仿真器上实现了一个用于无信号灯交叉口的统一控制系统。核心方法结合了分布式拍卖、基于纳什解的死锁求解、车队（platoon）感知的流量控制，并提供可由 DRL 调整的配置参数，适用于研究混合交通（单车 + 车队）协调策略。

核心思想
--------
- 使用去中心化拍卖引擎为参与者（单车或车队）生成优先级排序。
- 在冲突情况下用纳什定解器解决可能的死锁，生成 conflict_action（go/wait）。
- 交通控制器基于拍卖/纳什决策，调用 CARLA Traffic Manager 修改车辆行为（速度差、跟驰距离、忽视红绿灯/其它车辆比例等）。
- 支持对车队的整体控制：对 leader / follower 赋予不同参数以保持编队并提高通行效率。
- 以仿真时间记录车辆进入/退出控制、加速度样本等以便精确统计和后处理分析。

架构与主要模块
--------------
- CARLA（仿真环境）：世界、演员、交通管理器。
- ScenarioManager：场景设置、生成点、可视化调试区域、计时器。
- StateExtractor：每 tick 提取车辆状态（位置、速度、是否在路口等）。
- PlatoonManager：管理车队分组与导航信息。
- DecentralizedAuctionEngine：接受状态产生出价并返回有序胜者列表。
- DeadlockNashSolver：在冲突时间窗口内求解小规模博弈以避免死锁。
- TrafficController：将拍卖与纳什决策下发到 CARLA Traffic Manager，并记录历史与加速度数据。

数据流
-----
1. ScenarioManager 推进 CARLA 世界。
2. StateExtractor 提取当前车辆状态。
3. PlatoonManager 更新车队信息。
4. AuctionEngine 基于状态和车队信息计算出价与排序。
5. Nash 求解器处理冲突并建议 go/wait。
6. TrafficController 应用控制并采集统计数据。

方法论
-----
- 拍卖优先级：参与者根据策略或紧急程度提交出价，由分布式拍卖排序。
- 纳什死锁解决：对冲突子集建立博弈并求解稳定策略，减小死锁概率。
- 车队感知控制：leader 与 follower 分别使用不同 TM 参数，若车队成员已在路口中则保护其通行决策。
- 加速度过滤与统计：使用仿真时间采样，截断异常值并可选中值滤波，再计算平均加速度。

配置与 DRL
----------
- DRL 可调整参数集中管理（例如冲突时间窗口）；可在训练期间动态更新配置。
- 可选 max_go_agents（同时通行上限），用于对比实验；也可以设为 None 表示无限制。

运行方法
-------
前提：
- 安装并运行 CARLA server（对应客户端 egg）。
- 安装项目 python 依赖。

基本步骤：
1. 确保 CARLA egg 放在项目可被 main.py 找到的位置。
2. 启动 CARLA server。
3. 运行：
   python main.py
4. 控制台将周期性输出拍卖、车队与控制器状态与统计报告；程序结束时输出最终统计信息。

指标与日志
-----------
- TrafficController 记录：
  - 累计被控制车辆数、已离开车辆数、控制历史（使用仿真时间戳）
  - 每车正/负/绝对加速度样本（可滤波与截断异常）
- 推荐使用仿真时间作为主统计基准以保证可重复性。

可扩展性建议
--------------
- 将 DRL 训练循环与 DRLConfig 连接，实时更新参数并评估指标作为训练回报。
- 为 AuctionEngine 与 Nash 求解器增加更多诊断信息，便于训练奖励设计。
- 微调 traffic manager 参数以比较安全性与通过效率的权衡。

排错要点
--------
- 找不到 CARLA egg：检查 egg 路径或修改 main.py 的查找逻辑。
- 时间不一致：仿真统计优先使用 CARLA 的仿真时间戳。
- 车辆不响应：检查 actor id 是否存在、actor 是否 alive，以及 traffic manager 设置是否被正确调用。

许可证与贡献
-------------
- 建议添加 LICENSE（例如 MIT）和 CONTRIBUTING 指南。模块化设计便于单元测试和 DRL 集成。
