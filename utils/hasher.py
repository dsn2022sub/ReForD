import xxhash


def generate_hash(target):
    hasher = xxhash.xxh64()
    hasher.update(target)
    return str(hasher.intdigest())
