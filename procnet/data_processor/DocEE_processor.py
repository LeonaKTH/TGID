import logging
from typing import List, Dict
from pathlib import Path
import re
from procnet.data_processor.basic_processor import BasicProcessor
from procnet.conf.global_config_manager import GlobalConfigManager
from procnet.data_example.DocEEexample import DocEEDocumentExample, DocEEEntity, DocEELabel
from procnet.utils.util_data import UtilData
import json


class DocEEProcessor(BasicProcessor):
    def __init__(self, data_name):
        super().__init__()
        self.data_name = data_name
        self.data_path = GlobalConfigManager.get_dataset_path()
        logging.debug("Path: {}".format(self.data_path))

        self.train_path = self.data_path / data_name / "train.json"
        self.dev_path = self.data_path / data_name / "dev.json"
        self.test_path = self.data_path / data_name / "test.json"
        self.schema_path = self.data_path / data_name / "event_schema.json"
        self.duee_schema_path = self.data_path / data_name / "duee_fin_event_schema.json"
        logging.debug("train_path: {}".format(self.train_path))
        logging.debug("dev_path: {}".format(self.dev_path))
        logging.debug("test_path: {}".format(self.test_path))

        if data_name == 'ChFinAnn':
            self.train_json = UtilData.read_raw_json_file(self.train_path)
            self.dev_json = UtilData.read_raw_json_file(self.dev_path)
            self.test_json = UtilData.read_raw_json_file(self.test_path)

            self.train_docs: List[DocEEDocumentExample] = self.parse_json_all(self.train_json)
            self.dev_docs: List[DocEEDocumentExample] = self.parse_json_all(self.dev_json)
            self.test_docs: List[DocEEDocumentExample] = self.parse_json_all(self.test_json)

            self.SCHEMA = DocEELabel.EVENT_SCHEMA
            self.SCHEMA_KEY_ENG_CHN = DocEELabel.KEY_ENG_CHN
            self.SCHEMA_KEY_CHN_ENG = DocEELabel.KEY_CHN_ENG
        elif data_name == 'DuEE_Fin':
            self.train_json = UtilData.read_raw_json_file(self.train_path)
            self.test_json = UtilData.read_raw_json_file(self.test_path)

            self.train_docs: List[DocEEDocumentExample] = self.parse_json_all(self.train_json)
            self.dev_docs: List[DocEEDocumentExample] = self.parse_json_all(self.test_json)
            self.test_docs: List[DocEEDocumentExample] = self.parse_json_all(self.test_json)

            with open('Data/DuEE_Fin/duee_fin_event_schema.json', 'r', encoding='utf-8') as file:
                self.SCHEMA = {}
                for line in file.readlines():
                    one = json.loads(line)
                    self.SCHEMA[one['event_type']] = []
                    for item in one['role_list']:
                        self.SCHEMA[one['event_type']].append(item['role'])
        
            self.SCHEMA_KEY_ENG_CHN = None
            self.SCHEMA_KEY_CHN_ENG = None
        else:
            self.load_generic_dataset()

    def load_generic_dataset(self):
        dataset_dir = self.data_path / self.data_name
        if not dataset_dir.exists():
            raise FileNotFoundError("Dataset directory not found: {}".format(dataset_dir))
        if not self.train_path.exists():
            raise FileNotFoundError("train.json not found: {}".format(self.train_path))

        self.train_json = self.read_json(self.train_path)

        if self.dev_path.exists():
            self.dev_json = self.read_json(self.dev_path)
        elif self.test_path.exists():
            logging.warning("dev.json not found for %s, using first test sample as dev set", self.data_name)
            self.dev_json = self.read_json(self.test_path)[:1]
        else:
            logging.warning("dev.json and test.json not found for %s, using first train sample as dev set", self.data_name)
            self.dev_json = self.train_json[:1]

        if self.test_path.exists():
            self.test_json = self.read_json(self.test_path)
        else:
            logging.warning("test.json not found for %s, using dev.json as test set", self.data_name)
            self.test_json = self.dev_json

        self.train_docs = self.parse_json_all(self.train_json)
        self.dev_docs = self.parse_json_all(self.dev_json)
        self.test_docs = self.parse_json_all(self.test_json)

        self.SCHEMA = self.load_schema()
        self.SCHEMA_KEY_ENG_CHN = None
        self.SCHEMA_KEY_CHN_ENG = None

    @staticmethod
    def read_json(path: Path):
        data = UtilData.read_raw_json_file(path)
        if DocEEProcessor.is_raw_docee_data(data):
            logging.info("Converting raw DocEE format from %s", path)
            return DocEEProcessor.convert_raw_docee_data(data)
        return data

    @staticmethod
    def is_raw_docee_data(data):
        if not isinstance(data, list) or len(data) == 0:
            return False
        first = data[0]
        return (
            isinstance(first, list)
            and len(first) == 4
            and isinstance(first[0], str)
            and isinstance(first[1], str)
            and isinstance(first[2], str)
            and isinstance(first[3], list)
        )

    @staticmethod
    def convert_raw_docee_data(data):
        converted = []
        skipped_args = 0
        duplicate_roles = 0
        role_conflicts = 0
        skipped_docs = 0

        for doc_index, one in enumerate(data):
            title, text, event_type, arguments = one
            if not isinstance(text, str):
                skipped_docs += 1
                continue
            title = "" if title is None else str(title)
            event_type = str(event_type)
            doc_id = "{}-{}".format(doc_index, title[:80])
            protected_spans = [
                (arg["start"], arg["end"])
                for arg in arguments
                if isinstance(arg, dict) and "start" in arg and "end" in arg
            ]
            sentences, sentence_offsets = DocEEProcessor.split_text_to_sentences(text, protected_spans)

            ann_mspan2dranges = {}
            ann_mspan2guess_field = {}
            event_dict = {}

            for arg in arguments:
                if not isinstance(arg, dict):
                    skipped_args += 1
                    continue
                if not all(key in arg for key in ("start", "end", "type", "text")):
                    skipped_args += 1
                    continue

                start = arg["start"]
                end = arg["end"]
                role = arg["type"]
                span = arg["text"]
                if text[start:end] != span:
                    skipped_args += 1
                    continue

                position = DocEEProcessor.global_span_to_sentence_span(start, end, sentence_offsets)
                if position is None:
                    skipped_args += 1
                    continue

                if span in ann_mspan2guess_field and ann_mspan2guess_field[span] != role:
                    role_conflicts += 1
                    continue

                ann_mspan2guess_field[span] = role
                ann_mspan2dranges.setdefault(span, []).append(position)

                if role in event_dict:
                    duplicate_roles += 1
                    continue
                event_dict[role] = span

            converted.append([
                doc_id,
                {
                    "sentences": sentences,
                    "ann_mspan2dranges": ann_mspan2dranges,
                    "ann_mspan2guess_field": ann_mspan2guess_field,
                    "recguid_eventname_eventdict_list": [
                        ["{}-event-0".format(doc_id), event_type, event_dict]
                    ],
                },
            ])

        if skipped_args:
            logging.warning("Skipped %d raw DocEE arguments with invalid spans/fields", skipped_args)
        if skipped_docs:
            logging.warning("Skipped %d raw DocEE documents with invalid text fields", skipped_docs)
        if duplicate_roles:
            logging.warning(
                "Dropped %d duplicate raw DocEE role values because the current event dict stores one value per role",
                duplicate_roles,
            )
        if role_conflicts:
            logging.warning(
                "Skipped %d raw DocEE mentions whose same text appears with conflicting roles",
                role_conflicts,
            )
        return converted

    @staticmethod
    def split_text_to_sentences(text, protected_spans):
        if len(text) == 0:
            return [""], [(0, 0)]

        cut_points = []
        for match in re.finditer(r"\n+|(?<=[.!?])\s+", text):
            cut = match.end()
            if cut <= 0 or cut >= len(text):
                continue
            if DocEEProcessor.is_inside_any_span(cut, protected_spans):
                continue
            cut_points.append(cut)

        sentences = []
        offsets = []
        start = 0
        for cut in cut_points + [len(text)]:
            if cut <= start:
                continue
            sentences.append(text[start:cut])
            offsets.append((start, cut))
            start = cut
        return sentences, offsets

    @staticmethod
    def is_inside_any_span(index, spans):
        for start, end in spans:
            if start < index < end:
                return True
        return False

    @staticmethod
    def global_span_to_sentence_span(start, end, sentence_offsets):
        for sentence_index, (sentence_start, sentence_end) in enumerate(sentence_offsets):
            if sentence_start <= start and end <= sentence_end:
                return [sentence_index, start - sentence_start, end - sentence_start]
        return None

    def load_schema(self):
        if self.schema_path.exists():
            return self.read_schema_json(self.schema_path)
        if self.duee_schema_path.exists():
            return self.read_schema_json_lines(self.duee_schema_path)

        logging.warning("No schema file found for %s, inferring schema from dataset", self.data_name)
        return self.infer_schema_from_data()

    @staticmethod
    def read_schema_json(schema_path: Path):
        schema = UtilData.read_raw_json_file(schema_path)
        if not isinstance(schema, dict):
            raise ValueError("Schema file must be a JSON object: {}".format(schema_path))
        return schema

    @staticmethod
    def read_schema_json_lines(schema_path: Path):
        schema = {}
        with open(schema_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                one = json.loads(line)
                schema[one['event_type']] = []
                for item in one['role_list']:
                    schema[one['event_type']].append(item['role'])
        return schema

    def infer_schema_from_data(self):
        schema = {}
        all_json = []
        all_json.extend(self.train_json if hasattr(self, 'train_json') else [])
        all_json.extend(self.dev_json if hasattr(self, 'dev_json') else [])
        all_json.extend(self.test_json if hasattr(self, 'test_json') else [])

        for one in all_json:
            data = one[1]
            for item in data.get('recguid_eventname_eventdict_list', []):
                event_type = item[1]
                event_dict = item[2]
                if event_type not in schema:
                    schema[event_type] = []
                for role_name in event_dict.keys():
                    if role_name not in schema[event_type]:
                        schema[event_type].append(role_name)

        if len(schema) == 0:
            raise ValueError("Cannot infer schema from dataset {}".format(self.data_name))

        for event_type in schema:
            schema[event_type] = sorted(schema[event_type])
        return schema
        
    def parse_json_all(self, json) -> List[DocEEDocumentExample]:
        docs = []
        for one in json:
            doc_id: str = one[0]
            data = one[1]
            sentences: List[str] = data['sentences']
            ann_mspan2dranges: Dict[str, List[list]] = data['ann_mspan2dranges']
            ann_mspan2guess_field: Dict[str, str] = data['ann_mspan2guess_field']
            recguid_eventname_eventdict_list = data['recguid_eventname_eventdict_list']

            assert len(ann_mspan2dranges) == len(ann_mspan2guess_field)
            entities = []
            for k, v in ann_mspan2dranges.items():
                entity = DocEEEntity(span=k, positions=v, field=ann_mspan2guess_field[k])
                entities.append(entity)

            events = []
            for x in recguid_eventname_eventdict_list:
                event = {'EventType': x[1]}
                for k, v in x[2].items():
                    event[k] = v
                events.append(event)

            doc = DocEEDocumentExample(doc_id=doc_id, sentences=sentences, entities=entities, events=events)

            for entity in doc.entities:
                for pos in entity.positions:
                    assert entity.span == doc.sentences[pos[0]][pos[1]:pos[2]]
            
            docs.append(doc)
        return docs
