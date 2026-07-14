"""Entity Relationship Extractor module."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import networkx as nx
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables.config import RunnableConfig

from ....rag.llm import LLM, LLMConfig
from ....type import MultimodalSample
from ....ux import progress
from ._prompts import DEFAULT_ER_EXTRACTION_PROMPT
from .output_parser import EntityExtractionOutputParser

_DEFAULT_TUPLE_DELIMITER = "<|>"
_DEFAULT_RECORD_DELIMITER = "##"
_DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
_DEFAULT_ENTITY_TYPES = ["ORGANIZATION", "PERSON", "LOCATION", "EVENT", "DATE"]


@dataclass
class NERExtractorConfig:
    llm: LLMConfig
    prompt: str | Path = DEFAULT_ER_EXTRACTION_PROMPT
    entity_types: List[str] = field(default_factory=lambda: _DEFAULT_ENTITY_TYPES)
    tuple_delimiter: str = _DEFAULT_TUPLE_DELIMITER
    record_delimiter: str = _DEFAULT_RECORD_DELIMITER
    completion_delimiter: str = _DEFAULT_COMPLETION_DELIMITER


class NERExtractor:
    def __init__(
        self,
        prompt: BasePromptTemplate,
        output_parser: BaseOutputParser,
        llm: LanguageModelLike,
        *,
        chain_config: RunnableConfig | None = None,
    ):
        """Extracts entities and relationships from text units using a language model.

        Args:
            prompt_builder (PromptBuilder): The prompt builder object used to construct the prompt for the language model.
            llm (LanguageModelLike): The language model used for entity and relationship extraction.
            chain_config (RunnableConfig, optional): The configuration object for the extraction chain. Defaults to None.

        """
        self._extraction_chain = prompt | llm | output_parser
        self._chain_config = chain_config

    @classmethod
    def from_config(cls, config: NERExtractorConfig) -> "NERExtractor":
        """Builds and returns an instance of EntityRelationshipExtractor from a configuration object.

        Parameters:
            config (EntityRelationshipExtractorConfig): The configuration object for the entity relationship extractor.

        Returns:
            EntityRelationshipExtractor: An instance of EntityRelationshipExtractor built from the configuration object.
        """
        prompt = config.prompt
        if os.path.exists(prompt):
            prompt_template = PromptTemplate.from_file(prompt)
        else:
            assert isinstance(prompt, str)
            prompt_template = PromptTemplate.from_template(prompt)

        output_parser = EntityExtractionOutputParser(
            tuple_delimiter=config.tuple_delimiter,
            record_delimiter=config.record_delimiter,
        )
        llm = LLM.from_config(config.llm)

        prompt_template = prompt_template.partial(
            completion_delimiter=config.completion_delimiter,
            tuple_delimiter=config.tuple_delimiter,
            record_delimiter=config.record_delimiter,
            entity_types=",".join(config.entity_types),
        )

        return cls(
            prompt=prompt_template,
            output_parser=output_parser,
            llm=llm,
        )

    def invoke(self, sample: MultimodalSample) -> nx.Graph:
        """Invoke the entity relationship extraction process on the text.

        Parameters:
            sample (MultimodalSample): The sample to extract entities and relationships from.

        Returns:
            nx.Graph: A networkx Graph object representing the extracted entities and relationships.
        """
        chunk_graph = self._extraction_chain.invoke(
            input={"input_text": sample.text},
            config=self._chain_config,
        )

        return chunk_graph

    def invoke_batch(self, samples: List[MultimodalSample]) -> List[nx.Graph]:
        """Invoke the entity relationship extraction process on a batch of samples.

        Parameters:
            samples (List[MultimodalSample]): The samples to extract entities and relationships from.

        Returns:
            List[nx.Graph]: A list of networkx Graph objects representing the extracted entities and relationships.
        """
        return [
            self.invoke(sample)
            for sample in progress(samples, desc="Extracting entities", unit="sample")
        ]
