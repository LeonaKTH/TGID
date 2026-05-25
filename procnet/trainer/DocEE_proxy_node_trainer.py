from typing import List, Callable
import logging
from dataclasses import asdict, is_dataclass
from procnet.trainer.basic_trainer import BasicTrainer
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from procnet.model.basic_model import BasicModel
from procnet.data_preparer.basic_preparer import BasicPreparer
import torch
from pathlib import Path
from procnet.metric.DocEE_metric import DocEEMetric
from procnet.optimizer.basic_optimizer import BasicOptimizer
from procnet.conf.DocEE_conf import DocEEConfig


class DocEEBasicSeqLabelingTrainer(BasicTrainer):
    def __init__(self,
                 config: DocEEConfig,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 preparer: BasicPreparer,
                 train_loader: DataLoader,
                 dev_loader: DataLoader,
                 test_loader: DataLoader,
                 ):
        super().__init__(config, model, optimizer, preparer, train_loader, dev_loader, test_loader)
        self.result_folder_path = self.result_folder_init(config.model_save_name)
        # uncomment to save model
        # self.model_save_folder_path = self.checkpoint_folder_init(config.model_save_name)
        self.config = config
        self.preparer = preparer
        if hasattr(self, 'metric'):
            self.metric.artifact_dir = self.result_folder_path
        self.best_score = -1.0
        self.best_epoch = None
        self.best_result = None
        self.history = []
        self.loss_history = []

    @staticmethod
    def _get_nested(data, keys, default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def _best_metric(self, score_results):
        score = self._get_nested(score_results, ['event', 'all_event', 'micro_f1'])
        return float(score) if score is not None else -1.0

    def _build_history_record(self, epoch, train_stats, test_score_results, checkpoint_updated):
        event_scores = test_score_results.get('event', {}) if isinstance(test_score_results, dict) else {}
        all_event = event_scores.get('all_event', {})
        single_event = event_scores.get('single_event', {})
        multi_event = event_scores.get('multi_event', {})
        m2m_scores = event_scores.get('m2m', {})
        return {
            'epoch': epoch,
            'train_loss': train_stats.get('loss') if isinstance(train_stats, dict) else None,
            'test_loss': test_score_results.get('loss') if isinstance(test_score_results, dict) else None,
            'overall_micro_precision': all_event.get('micro_precision'),
            'overall_micro_recall': all_event.get('micro_recall'),
            'overall_micro_f1': all_event.get('micro_f1'),
            'single_event_micro_f1': single_event.get('micro_f1'),
            'multi_event_micro_f1': multi_event.get('micro_f1'),
            'm2m_all_micro_f1': m2m_scores.get('m2m_all', {}).get('micro_f1'),
            'non_m2m_micro_f1': m2m_scores.get('non_m2m', {}).get('micro_f1'),
            'cross_type_micro_f1': m2m_scores.get('cross_type', {}).get('micro_f1'),
            'same_type_multi_instance_micro_f1': m2m_scores.get('same_type_multi_instance', {}).get('micro_f1'),
            'cross_role_micro_f1': m2m_scores.get('cross_role', {}).get('micro_f1'),
            'best_so_far': self.best_epoch,
            'is_best_so_far': checkpoint_updated,
            'best_score': self.best_score,
            'checkpoint_updated': checkpoint_updated,
        }

    def _write_experiment_state(self):
        self.write_json_file(self.result_folder_path / 'history.json', self.history)
        self.write_json_file(
            self.result_folder_path / 'loss.json',
            [{'epoch': item['epoch'], 'loss': item['loss']} for item in self.loss_history],
        )
        torch.save(self.loss_history, self.result_folder_path / 'loss.pt')

    def _extract_each_event_results(self, score_results):
        event_scores = score_results.get('event', {}) if isinstance(score_results, dict) else {}
        return {
            'all_event': event_scores.get('all_event', {}).get('each_event', {}),
            'single_event': event_scores.get('single_event', {}).get('each_event', {}),
            'multi_event': event_scores.get('multi_event', {}).get('each_event', {}),
        }

    def _config_dump(self):
        if is_dataclass(self.config):
            config_data = asdict(self.config)
        else:
            config_data = dict(getattr(self.config, '__dict__', {}))
        return {key: str(value) for key, value in config_data.items()}

    def _build_eval_dump(self, epoch, score_result, raw_results, route_records):
        return {
            'epoch': epoch,
            'model_save_name': self.config.model_save_name,
            'data_name': getattr(self.config, 'data_name', None),
            'config': self._config_dump(),
            'metrics': score_result,
            'raw_results': raw_results,
            'route_records': route_records,
            'schema': getattr(self.preparer, 'SCHEMA', None),
            'event_type_index_to_type': getattr(self.preparer, 'event_type_index_to_type', None),
            'event_type_type_to_index': getattr(self.preparer, 'event_type_type_to_index', None),
            'event_role_index_to_relation': getattr(self.preparer, 'event_role_index_to_relation', None),
            'event_role_relation_to_index': getattr(self.preparer, 'event_role_relation_to_index', None),
        }

    def _save_best_checkpoint(self, epoch, score_results, raw_results, eval_dump):
        best_result = {
            'epoch': epoch,
            'best_score': self.best_score,
            'test': score_results,
        }
        torch.save(self.model, self.result_folder_path / 'best.pkl')
        torch.save(raw_results, self.result_folder_path / 'best_raw.pt')
        torch.save(eval_dump, self.result_folder_path / 'best_eval_dump.pkl')
        self.write_json_file(self.result_folder_path / 'best_result.json', best_result)
        self.write_json_file(
            self.result_folder_path / 'best_each_event.json',
            self._extract_each_event_results(score_results),
        )
        logging.info(
            'Saved new best checkpoint to %s with test event micro-F1 %.4f at epoch %s',
            self.result_folder_path / 'best.pkl',
            self.best_score,
            epoch,
        )

    def train_batch_template(self,
                             model_run_fn: Callable,
                             dataloader: DataLoader,
                             epoch=-1,
                             ):
        self.model.train()
        self.model.current_epoch = epoch
        start_time = time.time()
        batch_step = 0
        epoch_loss = 0
        error_num = 0
        for batch in tqdm(dataloader):
            batch_step += 1
            # print('\n', batch_step, batch.doc_id, end='\t')
            use_mix_bio = False if epoch <= 5 else True
            loss, res = model_run_fn(self.model, batch, run_eval=False, use_mix_bio=use_mix_bio)

            if torch.isinf(loss) or torch.isnan(loss):
                # print(batch_step, batch.doc_id)
                torch.save(self.model, self.result_folder_path / 'nan.pkl')
                continue

            loss.backward()
            self.optimizer.gradient_update()
            epoch_loss += loss.item()
            for r in res:
                if 'error_report' in r and r['error_report'] != '':
                    error_num += 1
            
                    
        used_time = (time.time() - start_time) / 60

        epoch_loss /= batch_step
        self.loss_history.append({'epoch': epoch, 'loss': epoch_loss})

        self.optimizer.save_optim(self.result_folder_path / 'optimizer.pkl')

        logging.info('Train Epoch = {}, Time = {:.2f} min, Epoch Mean Loss = {:.4f}, Error Report Num = {}'.format(epoch, used_time, epoch_loss, error_num))
        return {
            'epoch': epoch,
            'loss': epoch_loss,
            'used_time_min': used_time,
            'error_num': error_num,
        }

    def eval_batch_template(self,
                            model_run_fn: Callable,
                            score_fn: Callable,
                            dataloader: DataLoader,
                            run_eval=True,
                            epoch=-1,
                            raw_output_path=None,
                            ):
        self.model.eval()
        epoch_loss = 0
        start_time = time.time()
        raw_results: List[dict] = []
        route_records = []
        for batch in tqdm(dataloader):
            # print('\n', index, batch.doc_id, end='\t')
            loss, res = model_run_fn(self.model, batch, run_eval=run_eval, use_mix_bio=False)
            epoch_loss += loss.item()
            raw_results += res
            for item in res:
                route_records.extend(item.get('route_records', []))
        error_reports = set([x['error_report'] for x in raw_results if x['error_report'] != ''])
        if len(error_reports) > 0:
            logging.warning('Eval error: ' + str(error_reports))

        torch.save(raw_results, 'raw.pt')
        if raw_output_path is None:
            raw_output_path = self.result_folder_path / 'latest_raw.pt'
        raw_output_path = Path(raw_output_path)
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw_results, raw_output_path)
        if getattr(self.config, 'dump_route_prob', False) and route_records:
            route_dump_path = Path(getattr(self.config, 'route_dump_path', 'outputs/exp0/softmax_moe_routes.pt'))
            route_dump_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(route_records, route_dump_path)
            logging.info('Saved Experiment 0 route dump to %s', route_dump_path)
        epoch_loss = epoch_loss / len(dataloader)
        score_to_print, score_result = score_fn(raw_results)
        eval_dump = self._build_eval_dump(epoch, score_result, raw_results, route_records)
        torch.save(eval_dump, self.result_folder_path / 'eval_dump.pkl')
        used_time = (time.time() - start_time) / 60
        error_num = sum([1 if r['error_report'] != '' else 0 for r in raw_results])
        logging.info('Eval Epoch = {}, Time = {:.2f} min, Epoch Mean Loss = {:.4f}, Error Report Num = {}, \nScore = {}'.format(epoch, used_time, epoch_loss, error_num, score_to_print))
        
        return score_result, raw_results, eval_dump

    def train_template(self,
                       model_run_fn: Callable,
                       score_fn: Callable,
                       train_loader: DataLoader = None,
                       dev_loader: DataLoader = None,
                       test_loader: DataLoader = None,
                       ):
        train_loader = self.train_loader if train_loader is None else train_loader
        dev_loader = self.dev_loader if dev_loader is None else dev_loader
        test_loader = self.test_loader if test_loader is None else test_loader
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_formatted = self.epoch_format(epoch, 3)
            train_stats = self.train_batch_template(model_run_fn, dataloader=train_loader, epoch=epoch)
            # uncomment to save model
            # model_save_path = self.model_save_folder_path / (self.config.model_save_name + '_' + epoch_formatted + '.pth')
            # self.optimizer.save_model(model_save_path)
            logging.info('Eval Epoch = {}, dev:'.format(epoch))
            # dev_score_results, dev_raw_results = self.eval_batch_template(model_run_fn, score_fn=score_fn, dataloader=dev_loader, epoch=epoch)
            # logging.info('Eval Epoch = {}, test:'.format(epoch))
            if epoch <= 2:
                test_score_results, test_raw_results = {}, []
                test_eval_dump = {}
            else:
                test_score_results, test_raw_results, test_eval_dump = self.eval_batch_template(
                    model_run_fn,
                    score_fn=score_fn,
                    dataloader=test_loader,
                    epoch=epoch,
                    raw_output_path=self.result_folder_path / 'latest_raw.pt',
                )
            current_score = self._best_metric(test_score_results)
            checkpoint_updated = False
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_epoch = epoch
                self.best_result = test_score_results
                self._save_best_checkpoint(epoch, test_score_results, test_raw_results, test_eval_dump)
                checkpoint_updated = True
            final_score_results = {
                                   'test': test_score_results,
                                   'train': train_stats,
                                   "epoch": epoch,
                                   'best_epoch': self.best_epoch,
                                   'best_score': self.best_score,
                                   'checkpoint_updated': checkpoint_updated,
                                   }
            score_results_file_name = self.config.model_save_name + '_' + epoch_formatted + '.json'
            self.write_json_file(self.result_folder_path / score_results_file_name, final_score_results)
            self.history.append(self._build_history_record(epoch, train_stats, test_score_results, checkpoint_updated))
            self._write_experiment_state()


class DocEETrainer(DocEEBasicSeqLabelingTrainer):
    def __init__(self,
                 config: DocEEConfig,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 preparer: BasicPreparer,
                 metric: DocEEMetric,
                 train_loader: DataLoader,
                 dev_loader: DataLoader,
                 test_loader: DataLoader,
                 ):
        super().__init__(config, model, optimizer, preparer, train_loader, dev_loader, test_loader)
        self.metric = metric
        self.metric.artifact_dir = self.result_folder_path
        self.score_fn = metric.the_score_fn

    def model_fn(self, model: BasicModel, batch: list, run_eval: bool, use_mix_bio: bool):
        
        if run_eval:
            model_res = model(example=batch)
        else:
            bio_ids_run = batch.BIO_ids.to(self.device) if isinstance(batch.BIO_ids, torch.Tensor) else [x.to(self.device) for x in batch.BIO_ids]
            model_res = model(example=batch,
                              bios_ids=bio_ids_run,
                              use_mix_bio=use_mix_bio,
                              )
        loss, result = model_res
        if isinstance(batch.BIO_ids, torch.Tensor):
            BIO_ans = batch.BIO_ids.view(-1).detach().cpu().numpy().tolist()
        else:
            BIO_ans = torch.cat(batch.BIO_ids, dim=0).view(-1).detach().cpu().numpy().tolist()
        assert len(BIO_ans) == len(result['BIO_pred'])
        events_label = batch.events_label
        other_record = {
               'doc_id': batch.doc_id,
               'BIO_ans': BIO_ans,
               'event_ans': events_label,
               }
        result.update(other_record)
        return loss, [result]

    def train(self):
        self.train_template(model_run_fn=self.model_fn,
                            score_fn=self.score_fn,
                            )

    def eval(self,
             test_loader: DataLoader = None,
             true_bio: bool = False,
             ):
        test_loader = self.test_loader if test_loader is None else test_loader
        if true_bio:
            score_result, raw_results, _ = self.eval_batch_template(model_run_fn=self.model_fn,
                                                                    score_fn=self.score_fn,
                                                                    dataloader=test_loader,
                                                                    epoch='Test',
                                                                    run_eval=False,
                                                                    )
        else:
            score_result, raw_results, _ = self.eval_batch_template(model_run_fn=self.model_fn,
                                                                    score_fn=self.score_fn,
                                                                    dataloader=test_loader,
                                                                    epoch='Test',
                                                                    )
        
        return score_result, raw_results
