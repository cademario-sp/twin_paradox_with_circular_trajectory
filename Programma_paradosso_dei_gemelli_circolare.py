import numpy as np
import sys
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

# Costanti
g = 9.81        #Accelerazione in m/s^2
c = 2.99792458e8         # Velocità della luce in m/s
alpha_min_start = 1.3
#alpha_min = 1.194229584   # for 1e7 seconds
#alpha_min = 1.33207123828   # for 2e8 seconds (about 6 years)
#alpha_min = 1.5321775  # for 5e8 seconds (about 20 years)
#alpha_min = 0.0   # standard sin / cos acceleration
a_hat = g / c   # accelerazione g espressa in unità della velocità c
#alpha corrisponde al parametro che si utilizza per "far curvare" l'astronave

# Simulation parameters
max_iter = 15            # Numero massimo di iterazioni Newton-Raphson
#loop_time = 10000000.0  # Tempo trascorso sulla terra
loop_time = 100000000.0  # Tempo trascorso sulla terra
evaluations = 50000 
#Numero di volte nel moto circolare che si svolge il procedimento t_n -> t_n+1
tau_eval = np.linspace(0, loop_time, evaluations)  # Si stabiliscono nell'intervallo di tempo da 0 a loop_time le "evaluations" tali che essi siano equispaziati nell'intervallo di tempo 
tolerance=1e-4          # Visto che tutto è espresso in unità della velocità c,si tratta del valore in s/c che una volta raggiunta dice al programma di fermarsi

# Equazioni relativistiche del moto
def equations(alpha_min_param, tau, y):
    x_pos, y_pos, vx, vy, gamma = y
# Qui si spiega come il moto e la posizione dell'astronave variano secondo 3 parametri. Alpha_min_param è lo sterzo inizialmente applicato, tau è il tempo (visto secondo le evaluations) e f è la funzione che descrive la posizione x,y, la velocità x,y e l'istantaneo fattore gamma.

# Di seguito è riportato l’Ansatz per l’accelerazione:
#1)L’accelerazione in avanti (parallela) è simmetrica nel tempo tramite una funzione coseno:
#accelera nella prima metà del percorso e decelera in modo simmetrico nella seconda metà.
#Un fattore di scala α garantisce che all’inizio e alla fine l’accelerazione sia puramente parallela;
#2)l’accelerazione laterale verso il centro aumenta gradualmente (tramite una funzione seno),
#raggiungendo il valore massimo nel punto centrale, quindi diminuisce nello stesso modo fino a diventare zero
#alla fine del percorso. α può essere negativo ed è guidato da un parametro, α_min,
#ma con la condizione che 1.0 - alpha^2 cos^2(pi*tau/loop_time) >= 0

    alpha = 1 - alpha_min_param*np.sin( np.pi * tau / loop_time )
    alpha_times_cos = alpha * np.cos( np.pi * tau / loop_time )
    a_tangent_prop = a_hat * alpha_times_cos
    a_normal_prop = a_hat * np.sqrt( 1.0 - alpha_times_cos * alpha_times_cos )  # ensures that | a_prop | = a_hat
    
#Descrive come varia il paramentro di sterzo 

#    print(f": Acceleration components ({a_tangent_prop:.3e}) ({a_normal_prop:.3e}) Magnitude: ({c*np.sqrt(a_tangent_prop*a_tangent_prop + a_normal_prop*a_normal_prop ):3e})")

    v_squared = vx**2 + vy**2
    v_squared = min(v_squared, 0.999999)  # Evita che v >= c
    norm_v = np.sqrt(v_squared)
    gamma = 1.0 / np.sqrt(1 - v_squared)
    print(*["vx: ", vx, "vy: ", vy, "gamma: ", gamma])
    gamma_sqr = gamma * gamma
    gamma_cube = gamma_sqr * gamma

#   Trasformiamo a tang e norm in ax e ay del sistema di riferimento fisso (quello della "terra")
    ax = ((a_tangent_prop/gamma_cube) * vx - (a_normal_prop/gamma_sqr) * vy ) / norm_v
    ay = ((a_normal_prop/gamma_sqr) * vx + (a_tangent_prop/gamma_cube) * vy ) / norm_v

#Formule trovate su sezione "relativistic acceleration" Wikipedia

    return [vx, vy, ax, ay, gamma]

y0 = [0, 0, 1e-9, 0, 1.0]           # Funzione f parametro iniziale
alpha_min_param = alpha_min_start   # Parametro iniziale dello sterzo

for i in range(max_iter):
    sol = solve_ivp(lambda tau, y: equations(alpha_min_param, tau, y), [0, loop_time], y0,
                    t_eval=tau_eval, rtol=1e-12, atol=1e-12) #Il programma "risolve" automaticamente da sé l'equazione differenziale ordinaria
    xf, yf, vxf, vyf, invgamma = sol.y[:, -1]
    distance = np.sqrt( xf*xf + yf*yf )           # distanza dall'origine in t/c
    x_sol = sol.y[0] 
    y_sol = sol.y[1] 
    if distance < tolerance:
        print(f"Converged in {i+1} iterations.")
        break

    # Se la distanza è ancora maggiore della tolleranza voluta, si continua per minimizzare la distanza voluta. Fa uso del metodo Newton-Rhaftson. Secondo il quale si fa avvicinare un valore che somiglia alla derivata (la derivata non si può ottenere) a 0, dove la distanza è nulla
    h = 1e-14
    
    sol = solve_ivp(lambda tau, y: equations(alpha_min_param+h, tau, y), [0, loop_time], y0,
                    t_eval=tau_eval, rtol=1e-12, atol=1e-12)
    xf, yf, vxf, vyf, invgamma = sol.y[:, -1]
    distance_plus_h = np.sqrt( xf*xf + yf*yf )
    sol = solve_ivp(lambda tau, y: equations(alpha_min_param-h, tau, y), [0, loop_time], y0,
                    t_eval=tau_eval, rtol=1e-12, atol=1e-12)
    xf, yf, vxf, vyf, invgamma = sol.y[:, -1]
    distance_minus_h = np.sqrt( xf*xf + yf*yf )

    fprime = (distance_plus_h - distance_minus_h) / (2 * h)

    if fprime == 0:
        raise RuntimeError("Derivative vanished. Newton step impossible.")

    # Newton update
    alpha_min_param = alpha_min_param - distance / fprime
    print(f"new alpha_min_param and distance: ({alpha_min_param:.12e}, {distance:.12e})")
    #stabilisco un nuovo alpha... che farà diminuire le distanze

    # Plot trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(x_sol, y_sol, label="Relativistic Closed Loop")
    plt.scatter([x_sol[0]], [y_sol[0]], color='green', label='Start') 
    plt.scatter([x_sol[-1]], [y_sol[-1]], color='red', label='End')
    plt.xlabel("x [light-seconds]")
    plt.ylabel("y [light-seconds]")
    plt.title("Closed Relativistic Trajectory with Constant Proper Acceleration")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Output final error
xf, yf, vxf, vyf, invgamma = sol.y[:, -1]
print(f"Final position error : ({xf*c:.3e}, {yf*c:.3e})")
print(f"Final velocity error : ({vxf*c:.3e}, {vyf*c:.3e})")
sum_inv_gamma = np.sum(1.0 - (np.square(sol.y[2])+np.square(sol.y[3])))
time_spaceship = (loop_time/evaluations)*sum_inv_gamma #il tempo trascorso sull'astronave è la somma delle evaluazioni con le loro rispettive dilatazioni del tempo
print(f"total time in spaceship: ({time_spaceship:.3e})")
