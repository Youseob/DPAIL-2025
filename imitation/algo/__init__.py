from .ppo import PPO
from .gail import GAIL
from .dpail import DPAIL
from .infogail import InfoGAIL
from .bc import BC
ALGOS = {
    'gail': GAIL,
    'dpail': DPAIL,    
    'infogail': InfoGAIL,
    'bc': BC,
}
