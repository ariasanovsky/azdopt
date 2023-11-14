import os

import json

class Arguments:
    def __init__(
            self,
            parent_dir: str,
            num_blocks: int,
            dim: int = 2,
            tree: bool = False,
            project_name: str = None,
        ):
        self.parent_dir = parent_dir
        self.num_blocks = num_blocks
        self.dim = dim
        self.tree = tree
        self.project_name = project_name
    
    def __repr__(self):
        # print in order `num_blocks, dim, tree, parent_dir, project_name`
        return f"Arguments(num_blocks={self.num_blocks}, dim={self.dim}, tree={self.tree}, parent_dir='{self.parent_dir}', project_name='{self.project_name}')"
    
    def override_with(self, overrides):
        # overrides may have fields `dir, p, b, dim, t`
        if overrides.dir is not None:
            self.parent_dir = overrides.dir
        if overrides.p is not None:
            self.project_name = overrides.p
        if overrides.b is not None:
            self.num_blocks = overrides.b
        if overrides.dim is not None:
            self.dim = overrides.dim
        if overrides.t is not None:
            self.tree = overrides.t
    
    def project_path(self) -> str:
        return self.parent_dir + self.project_name
    
    def assign_last_project(self):
        # projects are in timestamp order
        if self.project_name is not None:
            return
        # get the last project
        projects = os.listdir(self.parent_dir)
        projects.sort()
        self.project_name = projects[-1]

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

