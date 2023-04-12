import os
import random
import numpy as np
from pathlib import Path

class SortOfARC:
    def __init__(self):
        self.shapes =  {0:  np.asarray([[0,1,0],[1,1,1],[0,1,0]]),
                        1:  np.asarray([[1,0,1],[0,1,0],[1,0,1]]),
                        2:  np.asarray([[0,0,1],[0,1,0],[1,0,0]]),
                        3:  np.asarray([[1,1,1],[0,1,0],[1,1,1]]),
                        4:  np.asarray([[1,1,1],[1,1,1],[1,1,0]]),
                        5:  np.asarray([[1,1,1],[1,1,1],[0,1,0]]),
                        6:  np.asarray([[1,1,1],[1,1,1],[0,1,1]]),
                        7:  np.asarray([[1,1,1],[1,0,1],[1,1,1]]),
                        8:  np.asarray([[1,0,1],[1,0,1],[1,1,1]]),
                        9:  np.asarray([[1,1,1],[1,0,1],[1,0,1]]),
                        10: np.asarray([[1,0,1],[0,1,1],[1,1,1]]),
                        11: np.asarray([[1,1,1],[0,1,1],[1,1,1]]),
                        12: np.asarray([[1,1,1],[1,1,0],[1,1,1]]),
                        13: np.asarray([[1,1,0],[1,1,1],[1,1,0]]),
                        14: np.asarray([[1,1,0],[1,1,1],[0,1,1]]),
                        15: np.asarray([[0,1,1],[1,1,1],[1,1,0]])
                        }

        self.colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # 0~9 color, 10~25 shape
        self.cond_range = len(self.shapes) + len(self.colors)
        self.color_map = ['black', 'red', 'green', 'blue', 'yellow', 'purple', 'pink', 'olive', 'cyan', 'brown', 'gray']
        self.background_color = 0
        self.image_size = 20
        self.object_size = 3
        self.num_objects = 3
        self.support_size = 5

    def generate_episode(self):
        
        condition = random.randint(0, self.cond_range-1)
        
        transformation = random.choice(['left', 'right', 'up', 'down'])
        input_images, output_images = [], []

        io_num = 0
        while io_num < self.support_size + 1:
            input_image, objects = self.generate_image(condition)
            output_image = self.apply_transformation(objects, transformation, condition)
            if output_image is None:
                continue
            io_num += 1
            input_images.append(input_image)
            output_images.append(output_image)
            
        query_image = input_images[-1]
        query_output = output_images[-1]

        return input_images[:-1], output_images[:-1], query_image, query_output, condition, transformation

    def generate_image(self, cond):
        objects = []
        positions = self.generate_positions()
        
        is_shape = False
        is_color = False
        if cond < 10:
            is_color = True
        else: 
            is_shape = True
            cond -= 10
            
        ''' make sure at least one object corresponds to the condition'''
        hit = False
        for i in range(self.num_objects):
            shape = random.randint(0, len(self.shapes)-1)
            color = random.choice(self.colors[1:])
            if (is_shape and shape == cond) or (is_color and color == cond):
                hit = True
            position = positions[i]
            objects.append((shape, color, position))
        if not hit:
            objects = objects[:-1]
            if is_color:
                shape = random.randint(0, len(self.shapes)-1)
                objects.append((shape, cond, positions[-1]))
            elif is_shape:
                color = random.choice(self.colors[1:])
                objects.append((cond, color, positions[-1])  )              

        image = np.zeros((self.image_size, self.image_size), dtype=np.int32)
        for shape, color, position in objects:
            x, y = position
            image[x:x+self.object_size, y:y+self.object_size] = self.shapes[shape] * color

        return image, objects

    def generate_positions(self):
        positions = []
        while len(positions) < self.num_objects:
            position = (random.randint(0, self.image_size - self.object_size),
                        random.randint(0, self.image_size - self.object_size))
            if not any([self.overlaps(position, p) for p in positions]):
                positions.append(position)
        return positions

    def overlaps(self, position1, position2):
        x1, y1 = position1
        x2, y2 = position2
        return (abs(x1 - x2) < self.object_size) and (abs(y1 - y2) < self.object_size)

    def legal_pos(self, position):
        x, y = position
        if x < 0 or x > self.image_size - self.object_size:
            return False
        if y < 0 or y > self.image_size - self.object_size:
            return False
        return True

    def apply_transformation(self, objects, transformation, cond):
        is_shape = False
        is_color = False
        if cond < 10:
            is_color = True
        else: 
            is_shape = True
            cond -= 10
        
        positions = []
        for i in range(len(objects)):
            shape, color, pos = objects[i]
            positions.append(pos)
            if (is_shape and shape == cond) or (is_color and color == cond):            
                if transformation == 'left':
                    new_pos = (pos[0], pos[1]-1)
                elif transformation == 'right':
                    new_pos = (pos[0], pos[1]+1)
                elif transformation == 'up':
                    new_pos = (pos[0]-1, pos[1])
                elif transformation == 'down':
                    new_pos = (pos[0]+1, pos[1])
                if not self.legal_pos(new_pos):
                    return None
                objects[i] = (shape, color, new_pos)
                positions[-1] = new_pos
        
        '''detect transformation position conflict'''
        conflict = False
        for i in range(self.num_objects):
            for j in range(i+1, self.num_objects):
                if self.overlaps(positions[i], positions[j]):
                    conflict = True
        if conflict:
            return None
        
        new_image = np.zeros((self.image_size, self.image_size), dtype=np.int32)
        for shape, color, pos in objects:
            x, y = pos
            new_image[x:x+self.object_size, y:y+self.object_size] = self.shapes[shape] * color

        return new_image
           

train_path = Path('img/train')
test_path = Path('img/test')
task_num = 1100
sarc = SortOfARC()
for task_id in range(task_num):
    path = train_path if task_id < 1000 else test_path
    t_id = task_id % 1000
    f = open(os.path.join(path, str(t_id)+'.json'), 'w')
    inp, out, qi, qo, cond, tranf = sarc.generate_episode()
    f.write('{"train": [')
    for i in range(len(inp)):
        f.write('{"input": ')
        f.write(np.array2string(inp[i], separator=',', precision=int))
        f.write(', "output": ')
        f.write(np.array2string(out[i], separator=',', precision=int))
        f.write('}')
        if i != len(inp)-1:
           f.write(', ')
        else: f.write('], ')
    f.write('"test": [{"input": ')
    f.write(np.array2string(qi, separator=','))
    f.write(', "output": ')
    f.write(np.array2string(qo, separator=','))
    f.write('}]}')
        
    f.close()

