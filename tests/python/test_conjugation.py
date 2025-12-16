import unittest
from pathlib import Path
import shutil
import pycauset


class TestConjugation(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_conjugation_tmp")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        import gc

        gc.collect()
        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                pass

    def test_vector_conj_dot(self):
        v = pycauset.ComplexFloat64Vector(2)
        w = pycauset.ComplexFloat64Vector(2)
        vc = None
        vt = None
        vh = None

        try:
            v.set(0, 1 + 2j)
            v.set(1, 3 - 4j)
            w.set(0, 5 + 6j)
            w.set(1, 7 + 8j)

            expected_dot = (1 + 2j) * (5 + 6j) + (3 - 4j) * (7 + 8j)
            self.assertEqual(v.dot(w), expected_dot)

            vc = v.conj()
            expected_vdot = (1 - 2j) * (5 + 6j) + (3 + 4j) * (7 + 8j)
            self.assertEqual(vc.dot(w), expected_vdot)

            vt = v.T
            vh = v.H

            self.assertFalse(v.is_conjugated())
            self.assertFalse(vt.is_conjugated())
            self.assertTrue(vh.is_conjugated())
            self.assertTrue(vh.is_transposed())
        finally:
            if vh is not None:
                vh.close()
            if vt is not None:
                vt.close()
            if vc is not None:
                vc.close()
            v.close()
            w.close()

    def test_matrix_H_conjugate_transpose(self):
        m = pycauset.ComplexFloat64Matrix(2)
        mh = None
        try:
            m.set(0, 0, 1 + 2j)
            m.set(0, 1, 3 + 4j)
            m.set(1, 0, 5 - 6j)
            m.set(1, 1, 7 - 8j)

            mh = m.H
            self.assertTrue(mh.is_transposed())
            self.assertTrue(mh.is_conjugated())

            # (m^H)_{0,1} = conj(m_{1,0})
            self.assertEqual(mh.get_element_as_complex(0, 1), (5 + 6j))
            # (m^H)_{1,0} = conj(m_{0,1})
            self.assertEqual(mh.get_element_as_complex(1, 0), (3 - 4j))
        finally:
            if mh is not None:
                mh.close()
            m.close()

    def test_persistence_roundtrip_conjugated_vector(self):
        v = pycauset.ComplexFloat64Vector(3)
        v.set(0, 1 + 2j)
        v.set(1, 3 - 4j)
        v.set(2, -5 + 6j)

        vc = v.conj()
        self.assertTrue(vc.is_conjugated())

        path = self.test_dir / "vc.pycauset"
        pycauset.save(vc, path)
        loaded = pycauset.load(path)
        try:
            self.assertTrue(loaded.is_conjugated())
            self.assertEqual(loaded.get_element_as_complex(0), (1 - 2j))
            self.assertEqual(loaded.get_element_as_complex(1), (3 + 4j))
            self.assertEqual(loaded.get_element_as_complex(2), (-5 - 6j))
        finally:
            loaded.close()
            vc.close()
            v.close()
