"""Security properties of own-installation helpers; no remote I/O."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class LocalConfigurationTests(unittest.TestCase):
    def test_new_local_secrets_are_distinct_not_printed_and_never_overwritten(self):
        module=load('configure_local')
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder);(destination/'.env.local.example').write_bytes((ROOT/'.env.local.example').read_bytes())
            output=io.StringIO()
            with patch.object(module,'ROOT',destination),patch('sys.argv',['configure_local','--project','new-test-project']),patch.object(module.secrets,'token_hex',side_effect=['a'*64,'b'*64,'c'*64]),contextlib.redirect_stdout(output):
                module.main()
                before=(destination/'.env').read_bytes()
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):module.main()
            self.assertEqual((destination/'.env').read_bytes(),before)
            for synthetic in ['a'*64,'b'*64,'c'*64]:
                self.assertIn(synthetic,before.decode());self.assertNotIn(synthetic,output.getvalue())
            self.assertIn('PRESCRIPTIVE_NEWS_ENABLED=false',before.decode())

    def test_invalid_ports_do_not_create_file(self):
        module=load('configure_local')
        with tempfile.TemporaryDirectory() as folder,patch.object(module,'ROOT',Path(folder)),patch('sys.argv',['configure_local','--api-port','8501','--frontend-port','8501']),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            module.main()

    def test_user_helper_saves_hash_only_and_preserves_existing_file(self):
        module=load('configure_academic_users');password='synthetic-classroom-password'
        with tempfile.TemporaryDirectory() as folder,patch.object(module,'ROOT',Path(folder)),patch('sys.argv',['configure_academic_users','--users','fixture_teacher']),patch.object(module,'getpass',return_value=password),contextlib.redirect_stdout(io.StringIO()) as output:
            module.main();data=(Path(folder)/'.academic-demo-users.json').read_text()
            self.assertNotIn(password,data);self.assertIn('pbkdf2_sha256',data)
            self.assertNotIn('pbkdf2_sha256',output.getvalue())
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):module.main()
            self.assertEqual((Path(folder)/'.academic-demo-users.json').read_text(),data)

    def test_probe_refuses_author_services_before_any_connection(self):
        module=load('probe_academic_provider')
        with patch('sys.argv',['probe','--origin','https://alpha-engine-academic-api.onrender.com','--user','fixture_teacher','--confirm-own-service']),patch.object(module,'DemoClient') as client,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            module.main()
        client.assert_not_called()

if __name__=='__main__':unittest.main()
