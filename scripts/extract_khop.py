import os
from threading import Thread
from queue import Queue


def read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d


def read_quadruples(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def read_quadruples_as_graph(filename, entity_dict, relation_dict):
    g = {}
    for quadruple in read_quadruples(filename):
        s = entity_dict[quadruple[0]]
        r = relation_dict[quadruple[1]]
        o = entity_dict[quadruple[2]]
        t = int(quadruple[3])
        if s not in g:
            g[s] = []
        if o not in g:
            g[o] = []
        g[s].append((o, r, t))
        g[o].append((s, r, t))
    return g


def build_khop(g, source, k, base):
    l = []
    q = Queue()
    q.put((source, 0, 0))
    while not q.empty():
        v, rs, d = q.get()
        for u, r, t in g[v]:
            if u == source:
                continue
            r = rs * base + r
            l.append(f'{source}\t{u}\t{r}\t{t}')
            if d + 1 < k:
                q.put((u, r, d + 1))
    return l


def main():
    total_tasks = int(os.getenv('TOTAL_TASKS', '32'))
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
    dataset = os.getenv('DATASET', 'GitGraph_TE_vscode')
    k = int(os.getenv('K', '1'))
    root_path = f'./data/{dataset}'
    os.makedirs(f'{root_path}/{k}-hop', exist_ok=True)
    entity_path = os.path.join(root_path, 'entities.dict')
    relation_path = os.path.join(root_path, 'relations.dict')
    train_path = os.path.join(root_path, 'train.txt')
    entity_dict = read_dictionary(entity_path)
    relation_dict = read_dictionary(relation_path)
    g = read_quadruples_as_graph(train_path, entity_dict, relation_dict)
    for v in g.keys():
        if v % total_tasks == task_id:
            l = build_khop(g, v, k, len(relation_dict))
            with open(f'{root_path}/{k}-hop/{v}.txt', 'w') as f:
                f.write('\n'.join(l) + '\n')


if __name__ == '__main__':
    main()
