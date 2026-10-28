import pathlib
import struct
import zlib

archive = pathlib.Path('downloads/best_models.zip')
data = archive.read_bytes()
position = 0
print('FILE_BYTES', len(data), flush=True)
while data[position:position+4] == b'PK\x03\x04':
    fields = struct.unpack_from('<4s5H3I2H', data, position)
    flags, compression, crc, compressed_size, raw_size, name_length, extra_length = fields[2], fields[3], fields[6], fields[7], fields[8], fields[9], fields[10]
    name = data[position+30:position+30+name_length].decode('utf8')
    payload = position + 30 + name_length + extra_length
    print(name, 'flags', flags, 'compression', compression, 'compressed', compressed_size, 'complete', payload+compressed_size <= len(data), flush=True)
    if flags & 8 or payload + compressed_size > len(data):
        break
    if name.endswith('.json'):
        raw = data[payload:payload+compressed_size]
        if compression == 8:
            raw = zlib.decompress(raw, -15)
        if zlib.crc32(raw) & 0xffffffff != crc:
            raise RuntimeError('CRC mismatch')
        destination = pathlib.Path(name)
        destination.parent.mkdir(exist_ok=True, parents=True)
        destination.write_bytes(raw)
    position = payload + compressed_size
