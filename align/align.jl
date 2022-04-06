"""

```
align(x, y, xmass, ymass)
```

or

```
align!(xnew, x, y, xmass, ymass)
```

aligns two structures (sets of points in 3D space), represented by vectors of static vectors. Solves the "Procrustes" problem. Structures are expected to be of the same size, and the correspondence is assumed from the vector indices. 

The non-mutation version `align` returns a new array, corresponding to the 
`x` coordinates rotated and translated to minimize the RMSD relative to `y`. 

The mutation call `align!` updates the `xnew` array.

L. Martinez, Institute of Chemistry - University of Campinas, Apr 6, 2022.

"""

using StaticArrays
using LinearAlgebra

# Non-mutating
function align(x, y, xmass, ymass) 
    xnew = similar(x)
    align!(xnew, x, y, xmass, ymass)
    return xnew
end

# Mutating
@views function align!(
    xnew::AbstractVector{SVector{3,T}},
    x::AbstractVector{SVector{3,T}}, 
    y::AbstractVector{SVector{3,T}}, 
    xmass::AbstractVector{T}, 
    ymass::AbstractVector{T}
) where T

    # Computing centers of mass
    cmx = zero(SVector{3,T})
    cmy = zero(SVector{3,T})
    tot_mass_x = zero(T)
    tot_mass_y = zero(T)
    for i in eachindex(x)
        cmx += x[i] * xmass[i]
        cmy += y[i] * ymass[i]
        tot_mass_x += xmass[i]
        tot_mass_y += ymass[i]
    end
    cmx = cmx / tot_mass_x
    cmy = cmy / tot_mass_y

    # Translating both sets to the origin
    for i in eachindex(x)
        x[i] = x[i] - cmx
        y[i] = y[i] - cmy
    end

    # Computing the quaternion matrix
    q = zero(MMatrix{4,4,T,16})
    for i in eachindex(x)
        xm = y[i] - x[i]
        xp = y[i] + x[i]
        q[1,1] = q[1,1] + xm[1]^2 + xm[2]^2 + xm[3]^2
        q[1,2] = q[1,2] + xp[2]*xm[3] - xm[2]*xp[3]
        q[1,3] = q[1,3] + xm[1]*xp[3] - xp[1]*xm[3]
        q[1,4] = q[1,4] + xp[1]*xm[2] - xm[1]*xp[2]
        q[2,2] = q[2,2] + xp[2]^2 + xp[3]^2 + xm[1]^2
        q[2,3] = q[2,3] + xm[1]*xm[2] - xp[1]*xp[2]
        q[2,4] = q[2,4] + xm[1]*xm[3] - xp[1]*xp[3]
        q[3,3] = q[3,3] + xp[1]^2 + xp[3]^2 + xm[2]^2
        q[3,4] = q[3,4] + xm[2]*xm[3] - xp[2]*xp[3]
        q[4,4] = q[4,4] + xp[1]^2 + xp[2]^2 + xm[3]^2
    end
    q[2,1] = q[1,2]
    q[3,1] = q[1,3]
    q[3,2] = q[2,3]
    q[4,1] = q[1,4]
    q[4,2] = q[2,4]
    q[4,3] = q[3,4]          

    # Computing the eigenvectors 'v' of the q matrix
    v = LinearAlgebra.eigvecs(q)[:,1]

    # Compute rotation matrix
    u = zero(MMatrix{3,3,T,9})
    u[1,1] = v[1]^2 + v[2]^2 - v[3]^2 - v[4]^2
    u[1,2] = 2 * ( v[2]*v[3] + v[1]*v[4] )
    u[1,3] = 2 * ( v[2]*v[4] - v[1]*v[3] )
    u[2,1] = 2 * ( v[2]*v[3] - v[1]*v[4] )
    u[2,2] = v[1]^2 + v[3]^2 - v[2]^2 - v[4]^2
    u[2,3] = 2 * ( v[3]*v[4] + v[1]*v[2] )
    u[3,1] = 2 * ( v[2]*v[4] + v[1]*v[3] )
    u[3,2] = 2 * ( v[3]*v[4] - v[1]*v[2] )
    u[3,3] = v[1]^2 + v[4]^2 - v[2]^2 - v[3]^2      

    # Rotate vector x, will be stored in xnew, and restore input arrays
    for i in eachindex(x)
        xnew[i] = cmy + u * x[i]
        x[i] = x[i] + cmx
        y[i] = y[i] + cmy
    end

    return nothing
end



