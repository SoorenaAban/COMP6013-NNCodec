import time

from nncodec.codec import *
from nncodec.logger import *
from nncodec.performence_display import *

lorem_ipsum_1par = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a consectetur ligula. Nunc erat dolor, tristique sed sagittis quis, dignissim eget erat. Vivamus enim lorem, finibus sit amet maximus eget, condimentum sit amet massa. Fusce aliquet velit sit amet ex pretium, ut tincidunt dolor semper. Nulla pellentesque eget massa quis rhoncus. Curabitur maximus quis mauris vel sollicitudin. Integer tristique ut nisl sed consequat. Donec a ipsum ut sem cursus ullamcorper. Sed finibus, sapien id volutpat tempus, turpis odio placerat purus, sit amet scelerisque nibh sem a magna. Sed justo sem, facilisis at imperdiet eu, tincidunt vel quam. Ut id sollicitudin eros, sit amet bibendum tortor. Lorem ipsum dolor sit amet, consectetur adipiscing elit."

lorem_ipsum_5par = """

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse vehicula nisi lectus. Curabitur pretium ipsum massa, quis pulvinar nisl tristique at. Pellentesque porttitor libero at magna efficitur convallis nec et eros. Pellentesque at ullamcorper nisl. Nullam fringilla maximus libero, molestie volutpat est pellentesque et. Nullam posuere vestibulum sem, ut tempor mi pharetra nec. Integer lacinia vel metus sit amet ullamcorper. Praesent sed viverra diam. Duis malesuada, lorem non mollis laoreet, libero justo facilisis lorem, nec euismod eros neque vulputate est. Vestibulum in vehicula erat. Suspendisse iaculis eu massa et pretium. Vivamus faucibus nisl velit, sed pretium sem molestie eget.

Proin non dolor nisi. Sed et ultricies risus. Integer malesuada maximus posuere. Phasellus ut vestibulum enim, ac placerat leo. Proin nibh tortor, dictum at scelerisque vel, bibendum ut metus. Donec tempus dui a nisi mattis posuere. Etiam tristique augue et massa aliquet tristique. Quisque dictum maximus nibh non laoreet.

Praesent tempus ante iaculis tortor facilisis tincidunt. Morbi fringilla arcu sed neque porta feugiat. Sed varius ultrices orci, at faucibus nisl feugiat nec. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Sed ac erat orci. Nunc vel bibendum est. Curabitur feugiat, lorem at cursus placerat, libero est hendrerit magna, sed condimentum ipsum metus at velit. Duis maximus nulla vitae lorem rhoncus, eu mollis massa blandit. Phasellus suscipit tincidunt maximus. Pellentesque sodales ut nunc ac laoreet. Vivamus bibendum aliquam risus, et ultrices libero scelerisque non. Sed eu massa sed justo porttitor suscipit. Praesent urna massa, ultrices vel ligula non, scelerisque tempus mi. Ut mattis velit eros, nec ultricies nulla auctor nec.

Praesent sit amet scelerisque leo, nec congue ante. Fusce scelerisque, risus quis accumsan aliquet, arcu nisl dictum urna, vitae scelerisque tortor enim eu tellus. Nulla euismod eros ac ullamcorper finibus. Sed placerat lectus aliquam aliquam consectetur. Nunc vehicula turpis odio, vel accumsan mi varius sit amet. Sed molestie, velit pellentesque dignissim aliquet, diam enim rhoncus est, nec egestas tellus dolor ut odio. In semper lectus commodo sapien malesuada, eu elementum libero aliquet. Donec et semper purus. Proin quis euismod libero. Morbi fringilla, orci vitae consequat laoreet, ante leo viverra ex, non consectetur neque libero nec elit.

Pellentesque faucibus nisl ipsum, ut porta risus pharetra placerat. Praesent a metus ut orci accumsan pellentesque nec eu risus. Quisque vitae lectus finibus, ullamcorper enim nec, efficitur dui. Fusce tortor nisl, venenatis ut ultricies ut, scelerisque ut libero. Fusce vestibulum sapien sed elit tempus tempus interdum non nisi. In efficitur, augue eu ornare interdum, velit metus condimentum nunc, id pellentesque eros lacus ac lorem. In bibendum neque lorem, eu vestibulum sem blandit et. Maecenas tristique metus id ultrices suscipit. Fusce et mollis massa. Donec condimentum, felis ac lacinia ullamcorper, odio augue dignissim libero, et venenatis nibh mi id enim. Duis ultrices tincidunt quam, et iaculis velit tempor posuere. Quisque facilisis ante ut arcu vulputate, eu posuere enim volutpat. Curabitur id ligula id dui laoreet dignissim eget vel tortor. Donec tortor odio, condimentum et leo vel, venenatis congue neque. """

def main():
    print(lorem_ipsum_5par)
    
    lorem_ipsum_bytes = str.encode(lorem_ipsum_1par)
    print(f"Sized of original data: {len(lorem_ipsum_bytes)}")
    
    bCodec = TfCodecByteArithmetic()
    logger = Logger()
    logger.display_info = False
    compressed_data = CompressedModel.serialize(bCodec.compress(lorem_ipsum_bytes, 0, logger=logger))
    print(f"Size of compressed data: {len(compressed_data)}")
    decompressed_data = bCodec.decompress(CompressedModel.deserialize(compressed_data), logger=logger)
    print(f"Size of decompressed data: {len(decompressed_data)}")
    
    if lorem_ipsum_bytes == decompressed_data:
        print("Data integrity preserved.")
    else:
        print("Data integrity compromised.")
        
    pm = PerformanceDisplay(logger.logs)
    pm.plot_coding_log()
    pm.plot_prediction_model_training_log()
    pm.plot_encoded_symbol_probability()
    
if __name__ == "__main__":
    main()