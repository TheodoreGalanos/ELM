from getpass import getpass
import os
os.environ['HF_TOKEN'] = getpass('Enter your HF token:')

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from abc import ABC, abstractmethod
from typing import Optional, Union, TypeVar, Generic

import numpy as np
from tqdm import trange

Phenotype = Optional[np.ndarray]
Mapindex = Optional[tuple]


# This Genotype class is implemented in Herbie's cleanup branch. Need to delete if you merge.
class Genotype(ABC):
    def __str__(self) -> str:
        raise NotImplementedError


GenoType = TypeVar('GenoType', bound=Genotype)


class MAPElites:
    def __init__(self, env, n_bins: int):
        self.env = env
        self.n_bins = n_bins
        # self.history_length = history_length
        self.history = []

        # discretization of behaviour space
        self.bins = np.linspace(*env.behaviour_space, n_bins + 1)[1:-1].T
        # perfomance of niches
        self.fitnesses = np.full([n_bins] * env.behaviour_ndim, -np.inf)
        # niches' sources
        self.genomes = np.zeros(self.fitnesses.shape, dtype=object)
        # index over explored niches to select from
        self.nonzero = np.full(self.fitnesses.shape, False)

        # bad mutations that ended up with invalid output.
        self.recycled = []
        # self.recycled_count = 0

        # outdated elites
        self.old_elites = []

        # qd-score
        self.qd_score = 0

        print(f"MAP of size: {self.fitnesses.shape} = {self.fitnesses.size}")

    def to_mapindex(self, b: Phenotype) -> Mapindex:
        return None if b is None else tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))

    def random_selection(self) -> Mapindex:
        ix = np.random.choice(np.flatnonzero(self.nonzero))
        return np.unravel_index(ix, self.nonzero.shape)

    def search(self, initsteps: int, totalsteps: int, atol=1, batch_size=32):
        tbar = trange(int(totalsteps))
        max_fitness = -np.inf
        max_genome = None

        config = {'batch_size': batch_size}

        for n_steps in tbar:
            self.history.append(np.copy(self.genomes))

            if n_steps < initsteps:
                # Initialise by generating initsteps random solutions.
                # comment: here we can sample 1 layout out of each prompt, to initiate the map
                x = self.env.random(**config)
            else:
                # Randomly select an elite from the map
                map_ix = self.random_selection()
                x = self.genomes[map_ix]
                # Mutate the elite
                x = self.env.mutate(x, **config)

            # Now that `x` is a list, we put them into the behaviour space one-by-one.
            for individual in x:
                map_ix = self.to_mapindex(self.env.to_behaviour_space(individual))

                # if the return is None, the individual is invalid and is thrown into the recycle bin.
                # comment: we should keep infeasible designs here if we can; eventually we'd need a metric of how far they are from a valid design
                if map_ix is None:
                    # self.recycled[self.recycled_count % len(self.recycled)] = individual
                    # self.recycled_count += 1
                    self.recycled.append(individual)
                    continue

                self.nonzero[map_ix] = True

                f = self.env.fitness(individual)
                # If new fitness greater than old fitness in niche, replace.
                if f > self.fitnesses[map_ix]:
                    self.fitnesses[map_ix] = f
                    self.old_elites.append((n_steps, self.genomes[map_ix]))
                    self.genomes[map_ix] = individual
                # If new fitness is the highest so far, update the tracker.
                if f > max_fitness:
                    max_fitness = f
                    max_genome = individual

                    tbar.set_description(f'{max_fitness=:.4f} of "{str(max_genome)}"')
                # If best fitness is within atol of the maximum possible fitness, stop.
                if np.isclose(max_fitness, self.env.max_fitness, atol=atol):
                    break
            
            self.qd_score = np.sum(np.absolute(np.nan_to_num(self.fitnesses, neginf=0))
            tbar.set_description(f'QD Score for iteration {n_steps}: {self.qd_score=:.4f}"')

        return str(self.genomes[np.unravel_index(self.fitnesses.argmax(), self.fitnesses.shape)])

    def plot(self):
        import matplotlib
        from matplotlib import pyplot

        matplotlib.rcParams['font.family'] = 'Futura'
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.style.use('ggplot')

        ix = tuple(np.zeros(self.fitnesses.ndim - 2, int))
        print(ix)
        map2d = self.fitnesses[ix]
        print(f'{map2d.shape=}')

        pyplot.pcolor(map2d, cmap='inferno')
        pyplot.show()


class BaseEnvironment(ABC, Generic[GenoType]):
    def __init__(self) -> None:
        self.genotype_space: np.ndarray

    @abstractmethod
    def random(self, n_seed, **kwarg) -> list[GenoType]:
        raise NotImplementedError

    @abstractmethod
    def mutate(self, x: GenoType, **kwarg) -> list[GenoType]:
        raise NotImplementedError

    @abstractmethod
    def fitness(self, x: GenoType) -> float:
        raise NotImplementedError

    @abstractmethod
    def to_behaviour_space(self, x: GenoType) -> Phenotype:
        raise NotImplementedError

    @property
    def max_fitness(self) -> int:
        return 0

    @property
    # [starts, endings) of search intervals
    def behaviour_space(self) -> np.ndarray:
        return self.genotype_space

    @property
    def behaviour_ndim(self) -> int:
        return self.behaviour_space.shape[1]


# NOTE: random seed and config are not really used in the Architext class. Need to also put them into the right
# places in production code.

import os
import random
from abc import ABC
from math import log, e
import re
from omegaconf import DictConfig, OmegaConf
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
from shapely.affinity import scale
from shapely.ops import unary_union
import networkx as nx
from typing import List
from PIL import Image, ImageDraw


def draw_polygons(polygons, colors, im_size=(512, 512), b_color="white", fpath=None):
    image = Image.new("RGBA", im_size, color="white")  # Image.new("L", im_size, color="white")
    draw = ImageDraw.Draw(image)

    for poly, color, in zip(polygons, colors):
        xy = poly.exterior.xy
        coords = np.dstack((xy[1], xy[0])).flatten()
        draw.polygon(list(coords), fill=(0, 0, 0))

        # get inner polygon coordinates
        small_poly = poly.buffer(-1, resolution=32, cap_style=2, join_style=2, mitre_limit=5.0)
        if small_poly.geom_type == 'MultiPolygon':
            mycoordslist = [list(x.exterior.coords) for x in small_poly]
            for coord in mycoordslist:
                coords = np.dstack((np.array(coord)[:, 1], np.array(coord)[:, 0])).flatten()
                draw.polygon(list(coords), fill=tuple(color))
        elif poly.geom_type == 'Polygon':
            # get inner polygon coordinates
            xy2 = small_poly.exterior.xy
            coords2 = np.dstack((xy2[1], xy2[0])).flatten()
            # draw it on canvas, with the appropriate colors
            draw.polygon(list(coords2), fill=tuple(color))

            # image = image.transpose(Image.FLIP_TOP_BOTTOM)

    if (fpath):
        image.save(fpath, format='png', quality=100, subsampling=0)
        np.save(fpath, np.array(image))

    return draw, image


def calc_entropy(labels, base=None):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def get_value(dictionary, val):
    for key, value in dictionary.items():
        if val == key:
            return value

    return "value doesn't exist"


def get_key(dictionary, val):
    for key, value in dictionary.items():
        if val == value:
            return key

    return "key doesn't exist"


def find_intersections(seed_polygon, target_polygons):
    """
        A function that finds intersections between a seed polygon and a list of candidate polygons.

    Args:
        seed_polygon (shapely polygon): A shapely polygon.
        target_polygons (list): A list of shapely polygons.

    Returns:
        array: The intersection matrix between the seed polygon and all individual target polygons.
    """
    intersect_booleans = []
    for _, poly in enumerate(target_polygons):
        try:
            intersect_booleans.append(seed_polygon.intersects(poly))
        except:
            intersect_booleans.append(True)
    return intersect_booleans


def find_distance(seed_graph, target_graphs):
    """
        A function that finds intersections between a seed polygon and a list of candidate polygons.

    Args:
        seed_polygon (shapely polygon): A shapely polygon.
        target_polygons (list): A list of shapely polygons.

    Returns:
        array: The intersection matrix between the seed polygon and all individual target polygons.
    """
    distances = [nx.graph_edit_distance(seed_graph, graph) for graph in target_graphs]
    return distances


def eval_function(samples, prompts, prompt_types):
    semantic_accuracy = []
    reward = []
    # assuming a batch of layouts sampled from the model
    for prompt, layout, prompt_type in zip(prompts, samples, prompt_types):
        geom = []
        try:
            # get layout geometry
            spaces, _, polygons = extract_layout_properties(layout)
            for poly in polygons:
                poly = [x for x in poly if x != ['']]
                poly = [x for x in poly if '' not in x]
                geom.append(Polygon(np.array(poly, dtype=int)))

            # get geometric properties: centroids and vectors
            room_centroids = get_room_centroids(geom)
            vectors = get_room_vectors(geom, room_centroids)

            # get layout annotations based on number of rooms and location
            desc = []
            num_desc = num_rooms_annotation(spaces)
            desc.extend(list(set(flatten(num_desc))))
            loc_desc = location_annotations(spaces, vectors)
            desc.extend(list(set(flatten(loc_desc))))
            desc = [re.sub('_', ' ', d) for d in desc]

            # calculate semantic accuracy: number of generations that satisfy the prompt
            semantic_accuracy.append(prompt in desc)

            # calculate reward according to type of prompt: difference or distance
            type_reward = get_reward(prompt, spaces, desc, prompt_type)
            reward.append(type_reward)
        except:
            # what type of values should we put when the model fails to create a valid design?
            semantic_accuracy.append(-1)
            reward.append(-1)

    results = {'semantic_accuracy': semantic_accuracy, 'reward': reward}
    # results.append((semantic_accuracy, type_reward))
    return results


housegan_labels = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6,
                   "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
regex = re.compile(".*?\((.*?)\)")


class LocalGenerator:
    """
    An ad hoc implementation of generating hf outputs in the local machine. Just to make things run temporarily and
    may not be final in the repo.
    """

    def __init__(self, token, model_str='architext/gptj-162M'):
        """
        Parameters:
            token: hf token.
            model_str: (Optional) hf model string.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token = token
        self.tokenizer = AutoTokenizer.from_pretrained(model_str, use_auth_token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(model_str, use_auth_token=self.token).to(self.device)

    def __call__(self, prompt, batch_size=16, **kwargs):
        config = {'return_tensors': 'pt'}

        output = self.model.generate(**self.tokenizer(prompt, **config).to(self.device),
                                     num_return_sequences=batch_size, **kwargs)
        return self.tokenizer.batch_decode(output)


class ArchitextGenotype(Genotype):
    architext_colors = [[0, 0, 0], [249, 222, 182], [195, 209, 217], [250, 120, 128], [126, 202, 234], [190, 0, 198],
                        [255, 255, 255],
                        [6, 53, 17], [17, 33, 58], [132, 151, 246], [197, 203, 159], [6, 53, 17], ]

    end_token_str = '<|endoftext|>'

    def __init__(self, code: str, height: float, layout: Optional[str]):
        self.code = code

        end_index = layout.find(self.end_token_str)
        cut_off_index = end_index + len(self.end_token_str) if end_index != -1 else None
        self.layout = layout[:cut_off_index].strip()

        self.height = height
        self.valid = self.validate()

    def get_clean_layout(self) -> str:
        if (len(self.layout.split('[layout]')) > 1):
            clean_layout = self.layout.split('[layout]')[1].split('[prompt]')[0].split(', ')
        else:
            clean_layout = self.layout.split('[Layout]')[1].split('[prompt]')[0].split(', ')
        return clean_layout

    def get_spaces(self) -> list:
        clean_layout = self.get_clean_layout()
        spaces = [re.sub(r'\d+', '', txt.split(':')[0]).lstrip() for txt in clean_layout]
        return spaces

    def get_space_ids(self) -> list:
        spaces = self.get_spaces()
        space_ids = [get_value(housegan_labels, space) for space in spaces]
        return space_ids

    def get_coordinates(self) -> list:
        clean_layout = self.get_clean_layout()
        coordinates = [txt.split(':')[1] for txt in clean_layout if len(txt.split(':')) > 1]
        coordinates = [re.findall(regex, coord) for coord in coordinates]
        coordinates = [x for x in coordinates if x != []]
        return coordinates

    def get_polygons(self) -> list:
        coordinates = self.get_coordinates()
        rectangles = []
        polygons = []
        for coord in coordinates:
            rectangles.append([point.split(',') for point in coord])
        for rec in rectangles:
            rec = [x for x in rec if x != ['']]
            rec = [x for x in rec if '' not in x]
            polygons.append(Polygon(np.array(rec, dtype=int)))

        return polygons

    def gfa(self) -> str:
        polygons = self.get_polygons()
        gfa = np.sum(np.array([poly.area() for poly in polygons]))
        return gfa

    def __str__(self) -> str:
        return self.layout if self.valid else ""

    def validate(self) -> bool:
        try:
            res = self.hlff() + self.gfa_entropy()
            img = self.get_image()
        except:
            return False
        return isinstance(res, float)

    def adjacency_matrix(self):
        scaled_polygons = []
        for polygon in self.get_polygons():
            scaled_polygons.append(scale(polygon, 1.15, 1.15, origin=polygon.centroid))
        intersection_matrix = np.zeros((len(scaled_polygons), len(scaled_polygons)))
        for k, p in enumerate(scaled_polygons):
            intersection_matrix[:, k] = find_intersections(p, scaled_polygons)
        return intersection_matrix

    def create_node_dict(self):
        space_ids = self.get_space_ids()
        values = [get_key(housegan_labels, id_) for id_ in space_ids]
        keys = np.arange(len(space_ids))
        return dict(zip(keys, values))

    def get_labelled_graph(self) -> list:
        adj_matrix = self.adjacency_matrix()
        labels = self.create_node_dict()
        graph = nx.from_numpy_matrix(adj_matrix)
        nx.relabel.relabel_nodes(graph, labels, copy=False)
        return graph

    def hlff(self) -> float:
        # Quality - hlff
        joined = unary_union(self.get_polygons())  # need to add this property to individual
        surface_area = joined.length * self.height  #
        floor_area = joined.area
        hlff = (2 * floor_area + surface_area) / floor_area
        return -hlff

    def gfa_entropy(self) -> float:
        room_gfa = [rm.area for rm in self.get_polygons()]
        gfa_entropy = calc_entropy(room_gfa)
        return gfa_entropy

    def typology(self) -> int:
        spaces = self.get_spaces()
        # typologies: [1b1b, 2b1b, 2b2b, 3b1b, 3b2b, 3b3b, 4b1b, 4b2b, 4b3b, 4b4b]
        typologies = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3)]
        nbed = np.where(np.array(spaces) == 'bedroom')[0].shape[0]
        nbath = np.where(np.array(spaces) == 'bathroom')[0].shape[0]
        return typologies.index((nbed, nbath))

    def get_image(self):
        polygons = self.get_polygons()
        return draw_polygons(polygons, self.architext_colors)[1]

    def _repr_png_(self):
        return self.get_image().tobytes()


class Architext(BaseEnvironment):
    """
    This will try to mutate layouts using architext-FIM models.

    The Heat Loss Form Factor will be used as the quality metric, defined as:
    heat loss form factor = heat loss area / treated floor area, assuming that all area is treated.
    Numerically, this is calculated as: hllff = sum(surface_area) / floor_area

    The behavioral descriptors will be layout typology (measured by number of bedrooms and bathrooms) and the entropy
    of the floor area distribution across different spaces in the layout.
    """
    # Record different definitions of behaviour spaces in a dict. Feel free to add.
    behaviour_mode_spec = {'hlff_and_fae':
                               {'genotype_ndim': 2,
                                'genotype_space': np.array([[0, 3.25], [0, 12]]).T
                                }
                           }
    model_param = {'do_sample': True,
                   'num_beams': 1,
                   'max_length': 500}
    room_labels = ['bedroom1', 'kitchen', 'living_room', 'corridor', 'bathroom1']

    def __init__(self, seed: str, config: Union[str, dict, DictConfig], prompts: list, height: float,
                 inference_server=None, behaviour_mode='hlff_and_fae', model_param=model_param):
        """
        Parameters:
            seed: the seed layouts.
            config: the config file or dict.
            prompts: list of different prompts that can be attached to selected layouts.
            inference_server: (Optional) the address of the inference server: 'domain:port'. If None, load model locally.
        """
        self.seed = seed
        self.height = height
        self.np_rng = np.random.RandomState(seed=np.random.randint(1, 1e8))

        if isinstance(config, str):
            self.config = OmegaConf.load(config)
        elif isinstance(config, (dict, DictConfig)):
            self.config = DictConfig(config)
        else:
            raise ValueError

        self.prompts = prompts
        self.model_param = model_param

        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=self.config.seed)

        self.behaviour_mode = behaviour_mode
        self.genotype_ndim = self.behaviour_mode_spec[self.behaviour_mode]['genotype_ndim']
        self.genotype_space = self.behaviour_mode_spec[self.behaviour_mode]['genotype_space']

        self.inference_server = inference_server
        self.local_generator = LocalGenerator(os.environ['HF_TOKEN']) if self.inference_server is None else None

    def random(self, **kwargs) -> List[ArchitextGenotype]:
        """
        Sample layouts from the model by randomly selecting prompts.
        Returns:
            the generated layout in a string representation.
        """
        return [ArchitextGenotype(code='', layout=x, height=self.height) for x in
                self._get_layout(self.seed + random.choice(self.prompts), **self.model_param, **kwargs)]

    def mutate(self, x: Genotype, **kwargs) -> List[ArchitextGenotype]:
        # TODO: batch_size > 1 is not implemented yet!!
        lines = x.layout.split(', ')

        cut_off = np.random.randint(1, 3, size=1)[0]
        cut_off = min(cut_off, len(lines) - 1)
        new_prompt = lines[0] + ', ' + ', '.join(lines[1:cut_off]) + ", " + random.choice(self.room_labels) + ":"
        # print(new_prompt)
        return [ArchitextGenotype(code='', layout=x, height=self.height) for x in
                self._get_layout(new_prompt, **self.model_param, **kwargs)]

    '''
    def mutate(self, x: Genotype, **kwargs) -> List[ArchitextGenotype]:
        """
        Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation
        on spaces in the layout *only* 
        example (ignore newlines): 
        --------------------------------
        [prompt] a bedroom is adjacent to the kitchen 
        [layout] bedroom1: (194,91)(135,91)(135,47)(194,47), 
        living_room: (121,194)(47,194)(47,91)(106,91)(106,106)(121,106), 
        bathroom1: (179,121)(135,121)(135,91)(179,91), 
        bedroom2: (209,165)(135,165)(135,121)(209,121), 
        bathroom2: (165,209)(135,209)(135,165)(165,165), 
        bedroom3: (121,238)(47,238)(47,194)(121,194), 
        kitchen: (135,77)(106,77)(106,18)(135,18), 
        corridor: (121,209)(121,106)(106,106)(106,77)(135,77)(135,209) <|endoftext|>
        --------------------------------
        The transform will mask one or more spaces from the layout (instead of a random span). 
        The model can then be tasked with predicting the masked spaces.
        --------------------------------
        <prefix token>
        [prompt] a bedroom is adjacent to the kitchen 
        [layout] bedroom1: (194,91)(135,91)(135,47)(194,47), 
        living_room: (121,194)(47,194)(47,91)(106,91)(106,106)(121,106), 
        <suffix token>
        bedroom2: (209,165)(135,165)(135,121)(209,121), 
        bathroom2: (165,209)(135,209)(135,165)(165,165), 
        bedroom3: (121,238)(47,238)(47,194)(121,194), 
        kitchen: (135,77)(106,77)(106,18)(135,18), 
        corridor: (121,209)(121,106)(106,106)(106,77)(135,77)(135,209) 
        <mask token>
        --------------------------------
        """

        #this btw requires the NeoX tokenizer: tokenizer = neox_args.tokenizer and NeoX FIM models
        suffix_tok_id, prefix_tok_id, middle_tok_id = self.tokenizer.vocab_size - 1, self.tokenizer.vocab_size, self.tokenizer.vocab_size + 1

        #get full layout
        contents = self.tokenizer.detokenize(x)
        # parse spaces from the layout
        prompt, layout = contents.split("[layout]")
        spaces = layout.split(", ")
        spaces = [s.strip() for s in spaces]
        mutated_layouts = []
        for i in range(0, self.n_mutations):
            try:
                # sample a number of spaces to mask, 
                # mask at least one, keep at least one ?
                num_spaces = self.np_rng.randint(1, len(spaces)-1)
                # select what contiguous spaces to mask
                start_idx = self.np_rng.randint(0, len(spaces)-num_spaces)
                end_idx = start_idx + num_spaces
            except ValueError as e:
                # should probably pass this result to failed mutations, or maybe throw it away and retry idk
                raise e

            prefix = prompt + "[layout] " + ", ".join(spaces[:start_idx])
            #middle = ", ".join(spaces[start_idx:end_idx])
            suffix = ", ".join(spaces[end_idx:]) + " "

            suffix = np.array([suffix_tok_id, *tokenizer.tokenize(suffix)])
            prefix = np.array([prefix_tok_id, *tokenizer.tokenize(prefix)])
            #middle = np.array([middle_tok_id, *tokenizer.tokenize(middle)])

            new_layout = np.concatenate([
                prefix,
                suffix,
                middle_tok_id,
            ])

            mutated_layouts.append(new_layout)

        return [self.local_generator(layout, batch_size=1, **kwargs) for layout in mutated_layouts]
    '''

    def fitness(self, x: ArchitextGenotype) -> float:
        if x.valid:
            return x.hlff()
        else:
            return -np.inf

    def to_behaviour_space(self, x: ArchitextGenotype) -> Phenotype:
        if not x.valid:
            return None

        try:
            return np.array([x.gfa_entropy(), x.typology()])
        except:
            return None

    def to_string(self, x: ArchitextGenotype) -> str:
        return str(x)

    def _get_layout(self, full_prompt, batch_size=16, **kwargs) -> Genotype:
        if self.inference_server is None:
            return self.local_generator(full_prompt, batch_size=batch_size, **kwargs)
        else:
            # TODO: Implement this.
            raise NotImplementedError()

    @staticmethod
    def _has_valid_output(x: ArchitextGenotype) -> bool:
        return x.valid

    def _update_seed(self):
        """
        Update the random seed in `self.config.seed` using `self.rng`.
        """
        self.config.seed = int(self.rng.integers(0, 1e8))

    @property
    def max_fitness(self):
        return 0

    @property
    # [starts, endings) of search intervals
    def behaviour_space(self):
        return self.genotype_space

    @property
    def behaviour_ndim(self):
        return self.behaviour_space.shape[1]

def main():
    import pickle

    seed = ""
    # target = "bedroom1: (194,106)(165,106)(165,47)(194,47), living_room: (179,223)(106,223)(106,121)(165,121)(165,135)(179,135), bathroom1: (165,106)(135,106)(135,77)(165,77), bedroom2: (135,106)(91,106)(91,33)(135,33), bathroom2: (106,165)(77,165)(77,135)(106,135), bedroom3: (91,106)(77,106)(77,121)(47,121)(47,62)(91,62), kitchen: (209,194)(179,194)(179,135)(194,135)(194,121)(209,121), corridor: (194,135)(165,135)(165,121)(106,121)(106,135)(77,135)(77,106)(194,106) <|endoftext|>"
    prompts = ["[prompt] a house with seven rooms and a corridor [layout]",
               "[prompt] a bedroom is located in the east side of the house [layout]",
               "[prompt] a house with two bedrooms and one bathroom [layout]"]

    config = {'seed': 42, }
    env = Architext(seed, config, height=2.0, prompts=prompts)
    elites = MAPElites(env, n_bins=12)
    for i in range(10):
        print("Best image", elites.search(initsteps=4 if i == 0 else 0, totalsteps=8))
        with open(f'elites_ckpt_{i}', 'wb') as f:
            pickle.dump(elites, f)

if __name__ == '__main__':
    main()