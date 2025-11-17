import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface EntradaModelo {
  genero: string;
  campesino: boolean;
  estrato: number;
  nivel_educacion: string;
  discapacidad: boolean;
  tipo_formacion: string;
  victima_conflicto: boolean;
  tiempo_segundos: number;
  puntaje_eje: number;
  autoidentificacion_etnica: string;
  eje_final: string;
  nivel: string;
  promedio_lineas?: number | null;
  promedio_areas?: number | null;
}

export interface RespuestaModelo {
  probabilidad_exito: number;
  prediccion: number;
}

@Injectable({
  providedIn: 'root'
})
export class PrediccionService {

  private apiUrl = 'http://localhost:8001/predict';

  constructor(private http: HttpClient) { }

  predecir(datos: EntradaModelo): Observable<RespuestaModelo> {
    return this.http.post<RespuestaModelo>(this.apiUrl, datos);
  }
}
