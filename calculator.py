import math
import numpy as np
from sympy import symbols, Eq, solve, cos, sin, sqrt, pi, tan, atan, sign

def calculate_axis(gamma_h, gamma_v, rounding=True):
    """
    Calculates the axis of the astigmatic lens according to Wesemann et al. (1991).
    https://www.researchgate.net/publication/21371741
    :param gamma_h: The angle of the dark crescent when measured from the horizontal axis.
    :param gamma_v: The angle of the dark crescent when measured from the vertical axis.
    :return: The axis of the astigmatic lens as degrees. 
        Returns a list of two values that are perpendicular to each other, both values satisfy the equations.
    """
    a1 = -math.atan(0.5*(-1 *2.0*math.sqrt(math.tan(gamma_h)**2*math.tan(gamma_v)**2 + 0.25*math.tan(gamma_h)**2 + 0.5*math.tan(gamma_h)*math.tan(gamma_v) + 0.25*math.tan(gamma_v)**2) + math.tan(gamma_h) + math.tan(gamma_v))/(math.tan(gamma_h)*math.tan(gamma_v)))
    a2 = -math.atan(0.5*( 1 *2.0*math.sqrt(math.tan(gamma_h)**2*math.tan(gamma_v)**2 + 0.25*math.tan(gamma_h)**2 + 0.5*math.tan(gamma_h)*math.tan(gamma_v) + 0.25*math.tan(gamma_v)**2) + math.tan(gamma_h) + math.tan(gamma_v))/(math.tan(gamma_h)*math.tan(gamma_v)))
    if rounding:
        if a1 < 0:
            a1 += math.pi
        if a2 < 0:
            a2 += math.pi
        return [round(math.degrees(a1), 4), round(math.degrees(a2), 4)]
    return [math.degrees(a1), math.degrees(a2)]

def check_solution(alpha_deg, Dcyl, Dsph, measurements, dc=-1.5, e=0.012, tolerance=0.0001):
    """
    Checks if the solution is valid.
    :param alpha_deg: The axis of the astigmatic lens in degrees.
    :param Dcyl: The diopter of the cylindrical lens.
    :param Dsph: The diopter of the spherical lens.
    :param measurements: The measurements of the dark crescents.
    :param dc: The distance from the camera to the eye.
    :param e: The eccentricity of the light source to the camera.
    :param tolerance: The tolerance for the solution.
    """
    alpha = math.radians(alpha_deg)
    dcr_h, gamma_h, dcr_v, gamma_v = measurements
    exp1 = (dc**2/e**2 * ((Dsph - 1/dc)**2 + Dcyl*(Dsph + Dcyl/2 - 1/dc) * (1 + math.cos(2*alpha))))**(-0.5) - dcr_h * np.sign(Dcyl*math.cos(2*alpha) + 2*Dsph + Dcyl - 2/dc)
    exp2 = math.atan((Dcyl*math.sin(2*alpha)) / (Dcyl*math.cos(2*alpha) + 2*Dsph + Dcyl - 2/dc)) - gamma_h
    exp3 = (dc**2/e**2 * ((Dsph - 1/dc)**2 + Dcyl*(Dsph + Dcyl/2 - 1/dc) * (1 + math.cos(2*(alpha+pi/2)))))**(-0.5) - dcr_v * np.sign(Dcyl*math.cos(2*(alpha+pi/2)) + 2*Dsph + Dcyl - 2/dc)
    exp4 = math.atan((Dcyl*math.sin(2*(alpha+pi/2))) / (Dcyl*math.cos(2*(alpha+pi/2)) + 2*Dsph + Dcyl - 2/dc))- gamma_v

    if abs(exp1) < tolerance and abs(exp2) < tolerance and abs(exp3) < tolerance and abs(exp4) < tolerance:
        return True
    return False

def calculate_diopters(DCR_h_value, DCR_v_value, gamma_h_value, gamma_v_value, e_value=0.012, d_value=-1.5, check=True, rounding=True):
    """
    Calculates the diopters of the astigmatic lens.
    :param DCR_h_value: The diopter of the horizontal dark crescent.
    :param DCR_v_value: The diopter of the vertical dark crescent.
    :param gamma_h_value: The angle of the dark crescent when measured from the horizontal axis.
    :param gamma_v_value: The angle of the dark crescent when measured from the vertical axis.
    :param e_value: The eccentricity of the light source to the camera.
    :param d_value: The distance from the camera to the eye.
    :param check: If True, checks if the solution is valid.
    :param rounding: If True, rounds the solution to 6 decimal places.

    equations were obtained through dark magic and are not to be questioned (or they were obtained through sympy)
    """
    Dcyl, alpha, DCR_h, DCR_v, gamma_h, gamma_v, d, e= symbols('Dcyl alpha DCR_h DCR_v gamma_h gamma_v d e')
    axis_eq =  -atan(0.5*(1 *2.0*sqrt(tan(gamma_h)**2*tan(gamma_v)**2 + 0.25*tan(gamma_h)**2 + 0.5*tan(gamma_h)*tan(gamma_v) + 0.25*tan(gamma_v)**2) + tan(gamma_h) + tan(gamma_v))/(tan(gamma_h)*tan(gamma_v)))
    alphas = calculate_axis(gamma_h_value, gamma_v_value)
    if alphas[0] % 45 == 0:
        raise ValueError("Axis is divisible by 45 degrees, tangents are infinite.")
    constants = {d: d_value, e: e_value, DCR_h: DCR_h_value, DCR_v: DCR_v_value, gamma_h: gamma_h_value, gamma_v: gamma_v_value, alpha: axis_eq}

    Dcyl_sol = [-2.0*sqrt(0.5*DCR_h**2*e**2*cos(2.0*alpha)/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)) + 0.5*DCR_v**2*e**2*cos(2.0*alpha)/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)) - e**2*(0.125*DCR_h**4*cos(4.0*alpha) - 0.125*DCR_h**4 + DCR_h**2*DCR_v**2*cos(2.0*alpha)**2 - 0.25*DCR_h**2*DCR_v**2*cos(4.0*alpha) + 0.25*DCR_h**2*DCR_v**2 + 0.125*DCR_v**4*cos(4.0*alpha) - 0.125*DCR_v**4)**0.5/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha))),
                 2.0*sqrt(0.5*DCR_h**2*e**2*cos(2.0*alpha)/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)) + 0.5*DCR_v**2*e**2*cos(2.0*alpha)/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)) - e**2*(0.125*DCR_h**4*cos(4.0*alpha) - 0.125*DCR_h**4 + DCR_h**2*DCR_v**2*cos(2.0*alpha)**2 - 0.25*DCR_h**2*DCR_v**2*cos(4.0*alpha) + 0.25*DCR_h**2*DCR_v**2 + 0.125*DCR_v**4*cos(4.0*alpha) - 0.125*DCR_v**4)**0.5/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha))),
                -2.0*sqrt(0.5*DCR_h**2*e**2*cos(2.0*alpha)/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)) + 0.5*DCR_v**2*e**2*cos(2.0*alpha)/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)) + e**2*(0.125*DCR_h**4*cos(4.0*alpha) - 0.125*DCR_h**4 + DCR_h**2*DCR_v**2*cos(2.0*alpha)**2 - 0.25*DCR_h**2*DCR_v**2*cos(4.0*alpha) + 0.25*DCR_h**2*DCR_v**2 + 0.125*DCR_v**4*cos(4.0*alpha) - 0.125*DCR_v**4)**0.5/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha))),
                 2.0*sqrt(0.5*DCR_h**2*e**2*cos(2.0*alpha)/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)) + 0.5*DCR_v**2*e**2*cos(2.0*alpha)/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)) + e**2*(0.125*DCR_h**4*cos(4.0*alpha) - 0.125*DCR_h**4 + DCR_h**2*DCR_v**2*cos(2.0*alpha)**2 - 0.25*DCR_h**2*DCR_v**2*cos(4.0*alpha) + 0.25*DCR_h**2*DCR_v**2 + 0.125*DCR_v**4*cos(4.0*alpha) - 0.125*DCR_v**4)**0.5/(2.0*DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)**3 - DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)*cos(4.0*alpha) + DCR_h**2*DCR_v**2*d**2*cos(2.0*alpha)))]

    Dcyls = []
    for eq in Dcyl_sol:
        # calculate value of s
        eq = eq.subs(constants)
        Dcyls.append(eq)

    Dsph_sol = [0.5*(-Dcyl*d*cos(2.0*alpha) - Dcyl*d - sqrt((0.5*DCR_h**2*Dcyl**2*d**2*(cos(4.0*alpha) - 1.0) + 4.0*e**2)/DCR_h**2) + 2.0)/d,
                0.5*(-Dcyl*d*cos(2.0*alpha) - Dcyl*d + sqrt((0.5*DCR_h**2*Dcyl**2*d**2*(cos(4.0*alpha) - 1.0) + 4.0*e**2)/DCR_h**2) + 2.0)/d]
    solutions = []
    for alpha_value in alphas:
        constants[alpha] = math.radians(alpha_value)
        for Dcyl_value in Dcyls:
            Dsph_1 = Dsph_sol[0].subs(constants).subs({Dcyl:Dcyl_value})
            Dsph_2 = Dsph_sol[1].subs(constants).subs({Dcyl:Dcyl_value})
            solutions.append([alpha_value, Dcyl_value, Dsph_1])
            solutions.append([alpha_value, Dcyl_value, Dsph_2])
    # round the solutions to 4 decimal places
    if rounding:
        solutions = [[round(sol[0], 6), round(sol[1], 6), round(sol[2], 6)] for sol in solutions]
    if check:
        solutions = [sol for sol in solutions if check_solution(sol[0], sol[1], sol[2], [DCR_h_value, gamma_h_value, DCR_v_value, gamma_v_value], d_value, e_value)]

    return solutions

def forward(Dsph_value, Dcyl_value, alpha_value, d_value=-1.5, e_value=0.012):
    Dsph, Dcyl, alpha, DCR_h, DCR_v, gamma_h, gamma_v, d, e= symbols('Dsph Dcyl alpha DCR_h DCR_v gamma_h gamma_v d e')
    constants = {Dsph: Dsph_value, Dcyl: Dcyl_value, alpha: alpha_value, d: d_value, e: e_value}
    eq1 = Eq((d**2/e**2 * ((Dsph - 1/d)**2 + Dcyl*(Dsph + Dcyl/2 - 1/d) * (1 + cos(2*alpha))))**(-0.5), DCR_h * sign(Dcyl*cos(2*alpha) + 2*Dsph + Dcyl - 2/d))
    eq2 = Eq(atan((Dcyl*sin(2*alpha)) / (Dcyl*cos(2*alpha) + 2*Dsph + Dcyl - 2/d)), gamma_h)
    eq3 = Eq((d**2/e**2 * ((Dsph - 1/d)**2 + Dcyl*(Dsph + Dcyl/2 - 1/d) * (1 + cos(2*(alpha+pi/2)))))**(-0.5), DCR_v * sign(Dcyl*cos(2*(alpha+pi/2)) + 2*Dsph + Dcyl - 2/d))
    eq4 = Eq(atan((Dcyl*sin(2*(alpha+pi/2))) / (Dcyl*cos(2*(alpha+pi/2)) + 2*Dsph + Dcyl - 2/d)), gamma_v)
    eqs = [eq1, eq2, eq3, eq4]

    sol = solve([eq.subs(constants) for eq in eqs], (DCR_h, gamma_h, DCR_v, gamma_v))
    out = {'DCR_h': sol[DCR_h], 'gamma_h': sol[gamma_h], 'DCR_v': sol[DCR_v], 'gamma_v': sol[gamma_v]}

    return out


