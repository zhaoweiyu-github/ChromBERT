from .interface import RegulatorEmbInterface, CistromeCellEmbInterface, PromptsCistromInterface, ExpCellEmbInterface

class RegulatorInterfaceManager():
    def __init__(self,config,prompt_map):
        super().__init__()
        self.prompt_regulator_cache_file = config.prompt_regulator_cache_file
        self.meta_file = config.meta_file
        self.prompt_map = prompt_map
        self.interface = self._create_interface()

    def _create_interface(self):
        if self.prompt_regulator_cache_file is not None:
            return RegulatorEmbInterface(self.prompt_regulator_cache_file)
        else:
            return PromptsCistromInterface(self.meta_file, self.prompt_map)
            
    def get_prompt_item(self, build_region_index, regulator, seq_len):
            if isinstance(self.interface, RegulatorEmbInterface):
                return self.interface.get_emb(build_region_index, regulator)
            elif isinstance(self.interface, PromptsCistromInterface):
                return self.interface.regulator_parse_prompts(regulator, seq_len)
            else:
                raise ValueError("Invalid interface type.")

class CelltypeInterfaceManager():
    def __init__(self,config,prompt_map):
        super().__init__()
        self.prompt_celltype_cache_file = config.prompt_celltype_cache_file
        self.prompt_kind = config.prompt_kind
        self.meta_file = config.meta_file
        self.prompt_map = prompt_map
        self.interface = self._create_interface()
        
    def _create_interface(self):
        if self.prompt_kind == "cistrome":
            if self.prompt_celltype_cache_file is not None:
                return CistromeCellEmbInterface(self.prompt_celltype_cache_file)
            else:
                return PromptsCistromInterface(self.meta_file,self.prompt_map)
        elif self.prompt_kind == "expression":
            return ExpCellEmbInterface(self.prompt_celltype_cache_file)
            
    def get_prompt_item(self, build_region_index, cell, seq_len):   
            if isinstance(self.interface, CistromeCellEmbInterface):
                return self.interface.get_emb(build_region_index, cell)
            elif isinstance(self.interface, PromptsCistromInterface):
                return self.interface.cistrome_celltype_parse_prompts(cell, seq_len)
            elif isinstance(self.interface, ExpCellEmbInterface):
                return self.interface.get_emb(cell)
            else:
                raise ValueError("Invalid interface type.")
