from my_os import *
import pandas as pd
from scipy.optimize import curve_fit


class sc_interpreter_class:
    def __init__(self):
        self.data_dict = open_folder()
        self.log_file = open('Report/report_text.txt', 'w')
        self.curdir = os.curdir
        self.sort_data_dict()
        self.smoothed_report()
        self.find_coefficients()
        self.physical_values_report()
        self.log_file.close()

    def physical_values_report(self):
        self.df_Current['kA'] = self.df_Current['V'] * self.Rogovski
        self.df_Tektronix['kV'] = self.df_Tektronix['V'] * self.Tektronix_VD * 1.0e-3


    def find_coefficients(self):
        pic_index = find_peaks(-self.df_Current['V'].values, prominence=1, distance=100)[0]
        pic_time = self.df_Current['us'].values[pic_index]
        pic_volt = self.df_Current['V'].values[pic_index]
        noise_ind = np.argwhere(self.df_Current['us'].values < 0)
        noise = self.df_Current['V'].values[noise_ind].max()
        current_start_ind = np.argwhere(np.abs(self.df_Current['V'].values) > noise).min()
        self.current_start_time = self.df_Current['us'].values[current_start_ind]
        current_start_volt = self.df_Current['V'].values[current_start_ind]

        def my_exp(x, a, b):
            return -a * np.exp(-x / b)

        opt, err = curve_fit(my_exp, pic_time, pic_volt)
        time_to_approx = np.arange(pic_time[0], pic_time[-1], np.gradient(self.df_Current['us'].values).mean())
        Rogowski_to_approx = my_exp(time_to_approx, opt[0], opt[1])
        plt.plot(time_to_approx, Rogowski_to_approx, label='decay')
        plt.plot(pic_time, pic_volt, 'o', label='picks')
        plt.plot(self.current_start_time, current_start_volt, 'o', label='current start')
        plt.plot(self.df_Current['us'], self.df_Current['V'], label='Rogowski')
        plt.xlabel('t, us')
        plt.ylabel('signal, V')
        plt.legend()
        plt.grid()
        plt.savefig('Report/picks.png')
        plt.show()
        self.period = np.gradient(pic_time).mean()
        self.log_file.write(f'Period is {self.period:3.2e} us\n')
        self.period *= 1.0e-6
        self.I_sc = 2.0 * np.pi * self.Capacity * self.U_0 / self.period * 1.0e-3
        self.log_file.write(f'Short circuit current is {self.I_sc:3.2e} kA\n')
        self.L_sc = (1 / self.Capacity) * (self.period / 2.0 / np.pi) ** 2
        self.log_file.write(f'Short circuit inductance is {self.L_sc:3.2e} H\n')
        self.Rogovski = -self.I_sc / pic_volt[0]
        self.log_file.write(f'Rogowski coefficient is {self.Rogovski:3.2e} kA/V\n')
        self.rise_time = pic_time.min() - self.current_start_time
        self.log_file.write(f'Rise time is {self.rise_time:3.2e} us\n')
        self.decay_time = opt[1]
        self.log_file.write(f'Decay time is {self.decay_time:3.2e} us\n')
        self.resistance = 2 * self.L_sc / self.decay_time / 1.0e-6
        self.log_file.write(f'Resistance is {self.resistance:3.2e} Ohm\n')

    def sort_data_dict(self):
        self.U_0 = self.data_dict['info']['Value']['U_0']
        self.Capacity = self.data_dict['info']['Value']['Capacity']
        self.n_conv = self.data_dict['info']['Value']['n_conv']
        self.Tektronix_VD = self.data_dict['info']['Value']['Tektronix_VD']
        self.df_Current = pd.DataFrame({
            'us': self.data_dict['waveform']['time'],
            'V': self.data_dict['waveform']['Rogowski']
        })
        self.df_Current = self.df_Current.rolling(self.n_conv, min_periods=1).mean()
        self.df_Systron = pd.DataFrame({
            'us': self.data_dict['waveform']['time'],
            'V': self.data_dict['waveform']['Systron']
        })
        self.df_Tektronix = pd.DataFrame({
            'us': self.data_dict['waveform']['time'],
            'V': self.data_dict['waveform']['Tektronix']
        })
        self.df_Tektronix = self.df_Tektronix.rolling(self.n_conv, min_periods=1).mean()
        self.df_Trig_out = pd.DataFrame({
            'us': self.data_dict['waveform']['time'],
            'V': self.data_dict['waveform']['Trig_out']
        })

    def smoothed_report(self):
        plt.title('smoothed')
        plt.plot(self.df_Current['us'], self.df_Current['V'], label='Rogowski')
        plt.plot(self.df_Trig_out['us'], self.df_Trig_out['V'], label='4Quick trig')
        plt.plot(self.df_Systron['us'], self.df_Systron['V'], label='Main trig')
        plt.plot(self.df_Tektronix['us'], self.df_Tektronix['V'], label='Tektronix')
        plt.xlabel('t, us')
        plt.ylabel('signal, V')
        plt.legend()
        plt.grid()
        plt.savefig('Report/smoothed.png')
        plt.show()
