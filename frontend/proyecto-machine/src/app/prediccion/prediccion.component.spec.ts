import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PrediccionComponent } from './prediccion.component';

describe('PrediccionComponent', () => {
  let component: PrediccionComponent;
  let fixture: ComponentFixture<PrediccionComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [PrediccionComponent]
    });
    fixture = TestBed.createComponent(PrediccionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
