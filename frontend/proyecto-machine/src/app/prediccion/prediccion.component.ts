import { Component } from '@angular/core';
import { PrediccionService, EntradaModelo, RespuestaModelo } from '../services/prediccion.service';

@Component({
  selector: 'app-prediccion',
  templateUrl: './prediccion.component.html',
  styleUrls: ['./prediccion.component.css']
})
export class PrediccionComponent {

  form: EntradaModelo = {
    genero: 'Femenino',
    campesino: false,
    estrato: 2,
    nivel_educacion: 'Educación Media',
    discapacidad: false,
    tipo_formacion: 'Virtual',
    victima_conflicto: false,
    tiempo_segundos: 3600,
    puntaje_eje: 70,
    autoidentificacion_etnica: 'Ningún grupo étnico',
    eje_final: 'Análisis de Datos',
    nivel: 'Básico',
    promedio_lineas: null,
    promedio_areas: null
  };

  resultado?: RespuestaModelo;
  cargando = false;
  errorMsg: string | null = null;

  constructor(private predService: PrediccionService) {}

  onSubmit() {
    this.cargando = true;
    this.errorMsg = null;
    this.resultado = undefined;

    this.predService.predecir(this.form).subscribe({
      next: (res) => {
        this.resultado = res;
        this.cargando = false;
      },
      error: (err) => {
        console.error(err);
        this.errorMsg = 'Ocurrió un error al obtener la predicción.';
        this.cargando = false;
      }
    });
  }
}
