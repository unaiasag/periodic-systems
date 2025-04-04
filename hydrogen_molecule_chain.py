import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy


def qubit_hamiltonian(E, a, k, t1, t2):
    """
    Convierte la matriz del Hamiltoniano Tight-Binding en operadores de Pauli.

    Parámetros:
    - k: Punto en la primera zona de Brillouin.
    - t: Parámetro de hopping.
    - n: Número de sitios en la cadena.

    Retorna:
    - Hamiltoniano en términos de operadores de Pauli (Qiskit `SparsePauliOp`).
    """
    
    fermionic_terms = {}
    fermionic_terms['+_0 -_0'] = E
    fermionic_terms['+_1 -_1'] = E
    fermionic_terms['+_0 -_1'] = -0.5*t1 - t2*np.exp(-1j*k*a)
    fermionic_terms['+_1 -_0'] = -0.5*t1 - t2*np.exp(1j*k*a)

    # Convertir a operador de fermiones
    fermionic_op = FermionicOp(fermionic_terms, num_spin_orbitals=2, copy=True)

    # Mapear a operadores de qubits con Jordan-Wigner
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)
    
    return qubit_op

def compute_energy_vqe(E, a, k, t1, t2):
    """
    Calcula la energía de la cadena de hidrógeno en un punto k usando VQE.

    Parámetros:
    - k: Punto en el espacio recíproco.
    - t: Parámetro de hopping.
    - n: Número de sitios en la cadena.

    Retorna:
    - Energía calculada con VQE.
    """

    backend = AerSimulator()
    hamiltonian = qubit_hamiltonian(E, a, k, t1, t2)
    ansatz = EfficientSU2(num_qubits=hamiltonian.num_qubits, entanglement='full')
    #ansatz = RealAmplitudes(hamiltonian.num_qubits)
    x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
    #print(x0)
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    ansatz_isa = pm.run(ansatz)
    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        #estimator.options.default_shots = 10000
        res = minimize(
            cost_func,
            x0,
            args=(ansatz_isa, hamiltonian_isa, estimator),
            method="cobyla", options={'maxiter':200}
        )

    return res.fun

def classical_energy(E, t1, t2, k_values):

    energies_1 = []
    energies_2 = []
    for k in k_values:
        e1 = E + np.sqrt(t1**2 + t2**2 + 2*t1*t2*np.cos(k*a))
        e2 = E - np.sqrt(t1**2 + t2**2 + 2*t1*t2*np.cos(k*a))
        energies_1.append(e1)
        energies_2.append(e2)

    return energies_1, energies_2


# Definir parámetros
E = 0
a = 1        # Parámetro de red de la celda primitiva
t1 = 1.0      # Parámetro de hopping
t2 = 0.75
k_values = np.linspace(0, np.pi/a, 10)  # Puntos en la primera zona de Brillouin

# Calcular la banda electrónica con VQE
energies = []
for k in k_values:
    cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
    }
    energy = compute_energy_vqe(E, a, k, t1, t2) 
    energies.append(energy)

e1, e2 = classical_energy(E, t1, t2, k_values)

# Graficar las bandas electrónicas
plt.plot((a/np.pi)*k_values, energies, '-o', label="VQE")
plt.plot((a/np.pi)*k_values, e1, '-r', label="Classical")
plt.plot((a/np.pi)*k_values, e2, '-r')
plt.xlabel("Vector de onda k ($\pi/a$)")
plt.ylabel("Energía (eV)")
plt.title("Bandas Electrónicas de una Cadena de Hidrógeno (Tight Binding + VQE)")
plt.legend()
plt.grid()
plt.show()