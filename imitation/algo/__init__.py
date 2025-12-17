from .ppo import PPO
from .gail import GAIL
from .dpail import DPAIL
from .infogail import InfoGAIL
ALGOS = {
    'gail': GAIL,
    'dpail': DPAIL,    
    'infogail': InfoGAIL
}
