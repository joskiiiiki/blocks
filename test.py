# debug_gpu.py - Complete GPU diagnostic

import pygame
import moderngl
import numpy as np
import sys

def test_moderngl_context():
    """Test ModernGL context creation"""
    print("=" * 60)
    print("STEP 1: Testing ModernGL Context Creation")
    print("=" * 60)
    
    # Test standalone context
    print("\nTrying standalone context...")
    try:
        ctx = moderngl.create_standalone_context(require=330)
        print(f"✓ Standalone context created successfully")
        print(f"  Version: {ctx.version_code}")
        print(f"  Vendor: {ctx.info.get('GL_VENDOR', 'unknown')}")
        print(f"  Renderer: {ctx.info.get('GL_RENDERER', 'unknown')}")
        print(f"  Max texture size: {ctx.info.get('GL_MAX_TEXTURE_SIZE', 'unknown')}")
        return ctx
    except Exception as e:
        print(f"✗ Standalone context failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_texture_creation(ctx):
    """Test simple texture creation"""
    print("\n" + "=" * 60)
    print("STEP 2: Testing Simple Texture Creation")
    print("=" * 60)
    
    if ctx is None:
        print("✗ No context available")
        return False
    
    # Create simple texture data
    width, height = 64, 64
    
    # Method 1: Raw numpy data
    print("\nMethod 1: Raw numpy array")
    try:
        data = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
        print(f"  Data shape: {data.shape}")
        print(f"  Data size: {data.nbytes} bytes")
        
        texture = ctx.texture(
            size=(width, height),
            components=4,
            data=data.tobytes()
        )
        print(f"  ✓ Texture created: {texture}")
        print(f"    GL object: {texture.glo}")
        texture.release()
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Method 2: Pygame surface
    print("\nMethod 2: Pygame surface")
    try:
        pygame.init()
        surf = pygame.Surface((width, height), pygame.SRCALPHA, 32)
        surf.fill((255, 0, 0, 255))
        
        flipped = pygame.transform.flip(surf, False, True)
        
        # Try tobytes
        try:
            tex_data = pygame.image.tobytes(flipped, "RGBA")
            print(f"  Using tobytes")
        except:
            tex_data = pygame.image.tostring(flipped, "RGBA", False)
            print(f"  Using tostring (fallback)")
        
        print(f"  Data size: {len(tex_data)} bytes")
        
        texture = ctx.texture(
            size=(width, height),
            components=4,
            data=tex_data
        )
        print(f"  ✓ Texture created: {texture}")
        texture.release()
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_large_texture(ctx):
    """Test larger texture (like atlas)"""
    print("\n" + "=" * 60)
    print("STEP 3: Testing Large Texture (Atlas Size)")
    print("=" * 60)
    
    if ctx is None:
        print("✗ No context available")
        return False
    
    # Typical atlas size
    width, height = 256, 256
    
    print(f"\nCreating {width}x{height} texture...")
    try:
        data = np.zeros((height, width, 4), dtype=np.uint8)
        data[:, :, 3] = 255  # Full alpha
        
        # Add some pattern
        data[::16, :, 0] = 255  # Red lines
        data[:, ::16, 1] = 255  # Green lines
        
        print(f"  Data size: {data.nbytes} bytes ({data.nbytes / 1024:.1f} KB)")
        
        print(f"  Creating texture...")
        sys.stdout.flush()  # Force output
        
        texture = ctx.texture(
            size=(width, height),
            components=4,
            data=data.tobytes()
        )
        
        print(f"  ✓ Texture created successfully")
        print(f"    Size: {texture.size}")
        print(f"    Components: {texture.components}")
        texture.release()
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_async_texture(ctx):
    """Test if texture creation is blocking"""
    print("\n" + "=" * 60)
    print("STEP 4: Testing Async/Blocking Behavior")
    print("=" * 60)
    
    if ctx is None:
        print("✗ No context available")
        return
    
    import time
    
    width, height = 512, 512
    data = np.zeros((height, width, 4), dtype=np.uint8)
    
    print(f"\nCreating {width}x{height} texture...")
    print("If this hangs, the context may be waiting for something...")
    sys.stdout.flush()
    
    start = time.time()
    
    try:
        texture = ctx.texture(
            size=(width, height),
            components=4,
            data=data.tobytes()
        )
        
        elapsed = time.time() - start
        print(f"  ✓ Created in {elapsed*1000:.2f}ms")
        
        # Try to use the texture
        print("  Testing texture binding...")
        texture.use(0)
        print("  ✓ Texture can be bound")
        
        texture.release()
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ Failed after {elapsed:.2f}s: {e}")

if __name__ == '__main__':
    print("GPU Diagnostics for ModernGL + Pygame\n")
    
    ctx = test_moderngl_context()
    
    if ctx:
        if test_texture_creation(ctx):
            if test_large_texture(ctx):
                test_async_texture(ctx)
    
    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)
