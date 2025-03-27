// Hyperbolic geometry utilities for Poincaré disk visualization

// Complex number utilities
export function complex(re, im) {
  return { re, im };
}

export function complexAdd(z1, z2) {
  return { re: z1.re + z2.re, im: z1.im + z2.im };
}

export function complexSub(z1, z2) {
  return { re: z1.re - z2.re, im: z1.im - z2.im };
}

export function complexMul(z1, z2) {
  return {
    re: z1.re * z2.re - z1.im * z2.im,
    im: z1.re * z2.im + z1.im * z2.re
  };
}

export function complexDiv(z1, z2) {
  const denom = z2.re * z2.re + z2.im * z2.im;
  return {
    re: (z1.re * z2.re + z1.im * z2.im) / denom,
    im: (z1.im * z2.re - z1.re * z2.im) / denom
  };
}

export function complexConj(z) {
  return { re: z.re, im: -z.im };
}

export function complexAbs(z) {
  return Math.hypot(z.re, z.im);
}

// Create evenly spaced values
export function linspace(start, end, num) {
  const step = (end - start) / (num - 1);
  return Array.from({ length: num }, (_, i) => start + i * step);
}

// Generate unit circle points
export function getUnitCircle(scale = 1.0, segments = 100) {
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    points.push(scale * Math.cos(angle));
    points.push(scale * Math.sin(angle));
  }
  return new Float32Array(points);
}

// Möbius transformation
export function mobiusTransform(z, a) {
  // (z - a) / (1 - conj(a) * z)
  return complexDiv(
    complexSub(z, a),
    complexSub(complex(1, 0), complexMul(complexConj(a), z))
  );
}

// Inverse Möbius transformation
export function inverseMobiusTransform(z, a) {
  // (z + a) / (1 + conj(a) * z)
  return complexDiv(
    complexAdd(z, a),
    complexAdd(complex(1, 0), complexMul(complexConj(a), z))
  );
}

// Create a straight line between two points
export function straightLine(z1, z2, numPoints = 20) {
  return linspace(0, 1, numPoints).map(t => ({
    re: z1.re + t * (z2.re - z1.re),
    im: z1.im + t * (z2.im - z1.im)
  }));
}

// Compute geodesic between two points in the Poincaré disk
export function getGeodesic(z1, z2, numPoints = 40) {
  try {
    // Normalize z1, z2 to use re/im properties
    z1 = { re: z1.re || z1.real || 0, im: z1.im || z1.imag || 0 };
    z2 = { re: z2.re || z2.real || 0, im: z2.im || z2.imag || 0 };
    
    // Handle special cases
    const z1Mag = Math.hypot(z1.re, z1.im);
    const z2Mag = Math.hypot(z2.re, z2.im);
    
    // Handle points near the origin specially
    if (z1Mag < 1e-6) {
      return straightLine(z1, z2, numPoints);
    }
    
    if (z2Mag < 1e-6) {
      return straightLine(z1, z2, numPoints);
    }
    
    // Calculate geodesic by mapping one point to the origin
    // Map z1 to origin
    const z2Prime = mobiusTransform(z2, z1);
    
    // Create a straight line in the transformed space
    const t = linspace(0, 1, numPoints);
    const transformedLine = t.map(val => ({
      re: val * z2Prime.re,
      im: val * z2Prime.im
    }));
    
    // Map back to the original disk
    return transformedLine.map(pt => inverseMobiusTransform(pt, z1));
  } catch (e) {
    console.error("Error in getGeodesic:", e);
    return straightLine(z1, z2, numPoints);
  }
}

// Circle inversion
export function circleInversion(z, c, R) {
  const zc = complexSub(z, c);
  const zcConj = complexConj(zc);
  const factor = complex(R * R / complexAbs(zc)**2, 0);
  const result = complexAdd(c, complexMul(factor, zcConj));
  return result;
}

// Hyperbolic distance
export function hyperbolicDistance(z1, z2) {
  const num = complexAbs(complexSub(z1, z2))**2;
  const den = (1 - complexAbs(z1)**2) * (1 - complexAbs(z2)**2);
  return Math.acosh(1 + 2 * num / den);
}
